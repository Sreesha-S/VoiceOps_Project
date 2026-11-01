"""Offline tests - need NO API keys.    python -m unittest discover -s tests -v
(Set DATASET_DIR in .env if the dataset folder is not next to this project.)"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from src import scoring as sc
from src.agent import SupportAgent
from src.calendar_tools import Calendar
from src.config import load_settings
from src.kb import load_all_chunks
from src.llm import LLMClient, LLMResponse
from src.orders import number_runs, parse_numbers
from src.retriever import Retriever
from src.voice import Voice, VoiceError, detect_language, speech_normalize, voice_turn

S = load_settings()
CAL = Calendar(S.dataset_dir, S.today)
CHUNKS = load_all_chunks(S.dataset_dir)
RETR = Retriever(CHUNKS)


class TestKnowledge(unittest.TestCase):
    def test_chunks_keep_ids_and_status(self):
        by = {c.chunk_id: c for c in CHUNKS}
        self.assertEqual(by["RETv3-2"].status, "superseded")
        self.assertIn("products.csv:ELE-0012", by)
        self.assertGreaterEqual(len([c for c in CHUNKS if c.doc_type != "catalog"]), 80)

    def test_sku_found_and_policy_beats_faq(self):
        ids = [c.chunk_id for c in RETR.search(["SKU ELE-0012 Assured claim"], k=5)]
        self.assertIn("products.csv:ELE-0012", ids)
        self.assertNotEqual(RETR.search(["how long do refunds take"], k=3)[0].chunk_id, "FAQ-1")


class TestDates(unittest.TestCase):
    def test_business_days(self):
        self.assertEqual(CAL.date_calc("add_business_days", "2026-10-17", 5)["result_iso"], "2026-10-24")
        self.assertEqual(CAL.date_calc("add_business_days", "2026-10-17", 7)["result_iso"], "2026-10-27")
        self.assertEqual(CAL.date_calc("add_calendar_days", "2026-10-13", 7)["result_iso"], "2026-10-20")


class TestNumbers(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(parse_numbers("Order number hai 8978 2837."), ("89782837", None))
        self.assertEqual(parse_numbers("Last four digits 4533 hain."), (None, "4533"))
        self.assertEqual(parse_numbers("आख़िरी चार अंक २९४९ हैं।"), (None, "2949"))
        self.assertEqual(parse_numbers("four five three three"), (None, "4533"))
        self.assertEqual(parse_numbers("last four digits 4533"), (None, "4533"))
        self.assertEqual(parse_numbers("refund started 17 October 2026"), (None, None))  # a date is not a code
        self.assertEqual(number_runs("pincode 440010"), ["440010"])


class TestScorer(unittest.TestCase):
    def test_scoring(self):
        self.assertTrue(sc.must_include_ok("It is due on the 27th of October 2026.", "2026-10-27 / 27 October 2026"))
        self.assertFalse(sc.must_include_ok("about a week", "5 ; 7"))
        self.assertFalse(sc.must_not_ok("takes 5 to 7 days", "5 to 7 days"))
        self.assertEqual(sc.parse_expected_calls("lookup_order(order_id=123, verification_value=4533) ; create_ticket()"),
                         [("lookup_order", {"order_id": "123", "verification_value": "4533"}), ("create_ticket", {})])


class FakeMsg(dict):
    def model_dump(self, exclude_none=True):
        return dict(self)


class TestLLMToolLoop(unittest.TestCase):
    def test_roundtrip_preserves_signature(self):
        call = {"id": "c1", "type": "function", "extra_content": {"google": {"thought_signature": "SIG"}},
                "function": {"name": "date_calc", "arguments": json.dumps({"operation": "add_business_days", "date_str": "2026-10-17", "n": 5})}}
        replies = [FakeMsg(role="assistant", content=None, tool_calls=[call]), FakeMsg(role="assistant", content="By 24 October 2026.")]
        seen = []

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kw):
                        seen.append(kw["messages"])
                        return NS(choices=[NS(message=replies.pop(0))])

        llm = LLMClient.__new__(LLMClient)
        llm.model, llm.effort, llm.client, llm.cache = "x", "", Fake, None
        out = llm.chat([{"role": "user", "content": "q"}], tools=CAL.schemas(), tool_funcs=CAL.functions())
        self.assertEqual(out.text, "By 24 October 2026.")
        self.assertEqual(out.tool_calls[0].result["result_iso"], "2026-10-24")
        second_request = seen[1]
        self.assertEqual(second_request[-2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "SIG")
        self.assertEqual(second_request[-1]["role"], "tool")


class TestRateLimits(unittest.TestCase):
    def test_pacing_between_real_calls(self):
        import time

        replies = [FakeMsg(role="assistant", content="a"), FakeMsg(role="assistant", content="b")]

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kw):
                        return NS(choices=[NS(message=replies.pop(0))])

        llm = LLMClient.__new__(LLMClient)
        llm.model, llm.effort, llm.client, llm.cache, llm.min_interval = "x", "", Fake, None, 0.3
        t0 = time.monotonic()
        llm.chat([{"role": "user", "content": "1"}])
        llm.chat([{"role": "user", "content": "2"}])
        self.assertGreaterEqual(time.monotonic() - t0, 0.29)

    def test_daily_quota_429_fails_fast(self):
        import src.llm as L

        class Quota(Exception):
            status_code = 429

        calls, slept = [], []
        orig = L.time.sleep
        L.time.sleep = lambda s: slept.append(s)
        try:
            with self.assertRaises(Quota):
                L.with_retries(lambda: calls.append(1) or (_ for _ in ()).throw(Quota("You exceeded your current quota")))
        finally:
            L.time.sleep = orig
        self.assertEqual(len(calls), 3)          # gave up after 3 tries, not 6
        self.assertLessEqual(sum(slept), 10)     # and did not wait minutes


class TestEmptyReplies(unittest.TestCase):
    def test_empty_answer_is_retried_and_not_cached(self):
        replies = [FakeMsg(role="assistant", content=""), FakeMsg(role="assistant", content="Real answer.")]

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kw):
                        return NS(choices=[NS(message=replies.pop(0))])

        class Cache(dict):
            def get(self, k):
                return dict.get(self, k)

            def put(self, k, v):
                self[k] = v

        llm = LLMClient.__new__(LLMClient)
        llm.model, llm.effort, llm.client, llm.cache = "x", "", Fake, Cache()
        self.assertEqual(llm.chat([{"role": "user", "content": "q"}]).text, "Real answer.")
        self.assertEqual(len(llm.cache), 1)
        self.assertEqual(list(llm.cache.values())[0]["content"], "Real answer.")

    def test_old_cached_empty_answer_is_ignored(self):
        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kw):
                        return NS(choices=[NS(message=FakeMsg(role="assistant", content="fresh"))])

        class Cache:
            def get(self, k):
                return {"role": "assistant", "content": ""}

            def put(self, k, v):
                pass

        llm = LLMClient.__new__(LLMClient)
        llm.model, llm.effort, llm.client, llm.cache = "x", "", Fake, Cache()
        self.assertEqual(llm.chat([{"role": "user", "content": "q"}]).text, "fresh")

    def test_number_words_count_as_digits(self):
        self.assertTrue(sc.must_include_ok("The courier makes up to three attempts.", "3"))
        self.assertTrue(sc.must_include_ok("the seventy-nine rupees express fee is credited", "79"))
        self.assertTrue(sc.must_include_ok("you get twenty one days", "21"))
        self.assertFalse(sc.must_include_ok("no one came", "3"))

    def test_product_category_pulls_in_the_return_rule(self):
        ids = [c.chunk_id for c in RETR.search(["The Comfi Cotton Briefs (Pack of 3) doesn't fit. Can I send it back?"], k=8)]
        self.assertIn("RET-5", ids)  # briefs are 'innerwear' in the catalog -> the 'cannot be returned' rule must appear

    def test_single_question_mode_always_has_escalation_rules(self):
        a = SupportAgent(ScriptedLLM(), RETR, CAL, S, mode="rag")
        ids = {c.chunk_id for c in a.standing}
        self.assertIn("ESC-1", ids)
        self.assertIn("KB-00.1", ids)


class ScriptedLLM:
    """Stands in for Gemini. Records what the agent sent; can 'decide' to call create_ticket."""
    model = "scripted"

    def __init__(self):
        self.last_user, self.reply, self.ticket_on = "", "OK.", None

    def chat(self, messages, *, tools=None, tool_funcs=None, max_rounds=6):
        if str(messages[0]["content"]).startswith("You turn a customer message"):
            last = messages[-1]["content"].splitlines()[-1]
            return LLMResponse(last)
        self.last_user = messages[-1]["content"]
        if self.ticket_on and tool_funcs and "create_ticket" in tool_funcs:
            res = tool_funcs["create_ticket"](summary=self.ticket_on, reason="human_requested", intent="test")
            return LLMResponse("A colleague will contact you." if res.get("created") else "Could not raise a ticket.")
        return LLMResponse(self.reply)


def agent_with(llm):
    return SupportAgent(llm, RETR, CAL, S, mode="conversation")


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.tickets = S.dataset_dir / "tools" / "tickets_created.csv"
        self.had_tickets = self.tickets.exists()

    def tearDown(self):
        if not self.had_tickets and self.tickets.exists():
            self.tickets.unlink()

    def test_verification_flow_and_privacy(self):
        llm = ScriptedLLM()
        c = agent_with(llm).start()
        r = c.respond("Hi, my order has not arrived")
        self.assertIn("No order ID yet", llm.last_user)
        r = c.respond("Order number 8978 2837")
        self.assertEqual([e.name for e in r.tool_calls], ["lookup_order"])
        self.assertNotIn("verification_value", r.tool_calls[0].args)
        self.assertIn("NOT verified", llm.last_user)
        self.assertIn("(none - customer not verified)", llm.last_user)
        r = c.respond("last four digits 1111")
        self.assertIn("attempt 1 of 2", llm.last_user)
        r = c.respond("sorry, it is 4533")
        self.assertTrue(c.desk.verified)
        self.assertEqual(r.tool_calls[0].args["verification_value"], "4533")
        self.assertEqual(r.tool_calls[0].public()["args"]["verification_value"], "****")  # masked in saved logs
        self.assertIn("VERIFIED", llm.last_user)
        self.assertIn("status:", llm.last_user)
        for secret in ("delivery_pincode", "customer_id", "4533"):
            self.assertNotIn(secret, llm.last_user.split("CONTEXT")[0].split("ORDER FACTS")[1])  # secrets never reach the model
        llm.reply = "Your digits 4533 match."
        self.assertNotIn("4533", c.respond("ok thanks").answer)  # echoed secret is stripped

    def test_lockout_after_two_failures(self):
        llm = ScriptedLLM()
        c = agent_with(llm).start()
        c.respond("order 8978 2837")
        c.respond("1111")
        c.respond("2222")
        self.assertTrue(c.desk.locked)
        r = c.respond("fine, then 4533")  # even the right value is now ignored
        self.assertFalse(c.desk.verified)
        self.assertIn("FAILED twice", llm.last_user)
        self.assertEqual(len([e for e in c.desk.events if e.name == "lookup_order"]), 3)  # 1 plain + 2 verification attempts

    def test_ticket_is_real_and_scrubbed(self):
        llm = ScriptedLLM()
        c = agent_with(llm).start()
        c.respond("order 8978 2837")
        c.respond("4533")
        llm.ticket_on = "Verified caller 4533 asks for a human about the delayed order; wants a person to decide."
        r = c.respond("I want to talk to a human")
        ev = [e for e in r.tool_calls if e.name == "create_ticket"][0]
        self.assertTrue(ev.result["created"])
        self.assertNotIn("4533", ev.args["summary"])
        self.assertTrue(r.ticket_id)
        self.assertTrue(any(i.startswith(("STY", "ESC", "IDV")) for i in r.chunk_ids))  # guides are always shown

    def test_short_ticket_fails_honestly(self):
        llm = ScriptedLLM()
        c = agent_with(llm).start()
        llm.ticket_on = "too short"
        self.assertEqual(c.respond("human please").answer, "Could not raise a ticket.")


class FakeSTT:
    def __init__(self, text):
        self.text, self.codes = text, []

    def transcribe(self, path, language_code):
        self.codes.append(language_code)
        return {"transcript": self.text}


class FakeTTS:
    def __init__(self, fail_languages=()):
        self.fail, self.calls = set(fail_languages), []

    def synthesize(self, text, **kw):
        self.calls.append((text, kw))
        if kw.get("language") in self.fail:
            raise RuntimeError("unsupported language")
        return b"RIFFfakewav"


class TestVoice(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "in.wav").write_bytes(b"x")

    def test_normalization_and_language(self):
        self.assertEqual(speech_normalize("Order 89782837 costs ₹1,499 **now**, arriving 2026-10-27."),
                         "Order 8978 2837 costs 1499 rupees now, arriving 27 October 2026.")
        self.assertEqual(detect_language("Aapka refund 5 din mein aa jayega, kya aap order ID bata sakte hain"), "hi-en")
        self.assertEqual(detect_language("आपका रिफंड जल्द आएगा"), "hi-IN")
        self.assertEqual(detect_language("Your refund will arrive soon."), "en-IN")

    def test_tts_language_fallback_chain(self):
        tts = FakeTTS(fail_languages={"hi-en"})
        v = Voice(S, stt=FakeSTT("x"), tts=tts)
        out = v.speak("Aapka refund aa jayega, kya aap order bata sakte hain", self.tmp / "o.wav")
        self.assertTrue(out.exists())
        self.assertEqual([c[1].get("language") for c in tts.calls], ["hi-en", "hi-IN"])
        self.assertEqual(tts.calls[0][1]["model"], S.tts_model)

    def test_tts_total_failure_is_reported(self):
        v = Voice(S, stt=FakeSTT("x"), tts=FakeTTS(fail_languages={"en-IN", "auto"}))
        v._tts.synthesize = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertRaises(VoiceError):
            v.speak("Hello", self.tmp / "o.wav")

    def test_unclear_audio_gets_polite_reprompt(self):
        v = Voice(S, stt=FakeSTT("   "), tts=FakeTTS())
        convo = agent_with(ScriptedLLM()).start()
        heard, reply, res, err = voice_turn(v, convo, self.tmp / "in.wav", self.tmp / "r.wav")
        self.assertEqual(heard, "")
        self.assertIn("say it again", reply)
        self.assertIsNone(res)
        self.assertTrue((self.tmp / "r.wav").exists())  # still answered with speech

    def test_full_voice_turn(self):
        llm = ScriptedLLM()
        llm.reply = "Your refund arrives by 24 October 2026."
        v = Voice(S, stt=FakeSTT("How long will my refund take?"), tts=FakeTTS())
        heard, reply, res, err = voice_turn(v, agent_with(llm).start(), self.tmp / "in.wav", self.tmp / "r.wav")
        self.assertEqual((heard, err), ("How long will my refund take?", ""))
        self.assertIn("CHANNEL: phone call", llm.last_user)
        self.assertEqual(v._stt.codes, [S.stt_language])


if __name__ == "__main__":
    unittest.main()
