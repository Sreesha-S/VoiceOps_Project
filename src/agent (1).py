"""The agents.

  PromptOnlyAgent  - STEP 1: a system prompt and nothing else (it will guess).
  SupportAgent     - STEPS 2 + 3. mode="rag"          -> answers single questions from the documents.
                                  mode="conversation" -> also remembers the call, verifies the caller,
                                                         looks up the order and hands over to a human.

One turn of SupportAgent (conversation mode):
  1. code reads order ID / verification digits from the message and calls lookup_order
  2. the LLM rewrites the message into a clean English search query (uses the history)
  3. search finds the best chunks; guide chunks (identity, escalation, style) are always included
  4. the LLM answers from CALL STATE + ORDER FACTS + CONTEXT, and may call date_calc / check_pincode / create_ticket
  5. we record exactly which chunks and tools were used
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import prompts
from .orders import TICKET_SCHEMA, OrderDesk, is_numbers_only, load_mock_tools

CLAIM = re.compile(r"ticket|टिकट|will contact|contact you|sampark karenge|संपर्क करेंगे|get in touch", re.I)
FALLBACK = "Sorry, I'm having trouble answering right now. Could you please say that again?"


@dataclass
class AgentResult:
    answer: str
    chunk_ids: list = field(default_factory=list)      # everything the LLM was shown
    retrieved_ids: list = field(default_factory=list)  # what search itself found
    rewritten_query: str = ""
    tool_calls: list = field(default_factory=list)     # list of ToolEvent
    ticket_id: str = ""


class PromptOnlyAgent:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, question: str) -> AgentResult:
        msgs = [{"role": "system", "content": prompts.STEP1_SYSTEM}, {"role": "user", "content": question}]
        return AgentResult(answer=self.llm.chat(msgs).text)


class SupportAgent:
    def __init__(self, llm, retriever, calendar, settings, mode: str = "conversation"):
        assert mode in ("rag", "conversation")
        self.llm, self.retriever, self.calendar, self.s, self.mode = llm, retriever, calendar, settings, mode
        self.tools = load_mock_tools(settings.dataset_dir) if mode == "conversation" else None
        standing = list(("KB-00.1", "KB-00.3"))
        guides = [c.chunk_id for c in retriever.chunks if c.doc_type == "agent_guide"]
        if mode == "conversation":  # identity, escalation and style rules are always in front of the model
            standing += guides
        else:  # single questions: still always know when a human must decide, and never leak other people's data
            standing += [g for g in guides if g.startswith("ESC") or g == "IDV-4"]
        self.standing = [retriever.by_id[i] for i in standing if i in retriever.by_id]

    def start(self) -> "Conversation":
        return Conversation(self)

    def answer(self, question: str) -> AgentResult:
        """One-off question = a brand-new conversation (used by the practice-question evaluation)."""
        return self.start().respond(question)


class Conversation:
    def __init__(self, agent: SupportAgent):
        self.a = agent
        self.history: list = []
        self.desk = OrderDesk(agent.tools, agent.calendar) if agent.mode == "conversation" else None

    # ---- step 2 of a turn
    def _rewrite(self, text: str) -> str:
        if is_numbers_only(text):
            return ""  # order IDs / verification digits need no document search
        parts = []
        if self.history:
            parts.append("Conversation so far:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-6:]))
        if self.desk and self.desk.verified:
            o = self.desk.order
            items = "; ".join(f"{i['title']} ({i['return_category']})" for i in o.get("items", []))
            parts.append(f"ORDER CONTEXT: items: {items}. placed {str(o.get('placed_at', ''))[:10]}, delivered {o.get('delivered_on') or 'not yet'}, "
                         f"status {o.get('status')}, payment {o.get('payment_method')}, delivery type {o.get('delivery_type')}.")
        parts.append(f"Customer message:\n{text}")
        try:
            q = self.a.llm.chat([{"role": "system", "content": prompts.REWRITE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]).text
        except Exception:  # noqa: BLE001 - a failed rewrite must never block the answer
            return text
        q = q.strip().splitlines()[0].strip(" \"'") if q.strip() else text
        return "" if q.upper() == "NONE" else q

    def _value_hint(self, shown) -> str:
        """The model must compare a product's price with the high-value threshold of ESC-1, and it does that unreliably.
        So our code does the comparison (the threshold is READ from ESC-1, not hardcoded) and passes the result on."""
        esc = self.a.retriever.by_id.get("ESC-1")
        m = re.search(r"more than ₹\s?([\d,]+)", esc.text) if esc else None
        if not m:
            return ""
        threshold, lines = int(m.group(1).replace(",", "")), []
        for c in shown:
            if c.doc_type != "catalog":
                continue
            price = re.search(r"price: Rs (\d+)", c.text)
            if price and int(price.group(1)) > threshold:
                lines.append(f"{c.text.split(' (SKU')[0]} (SKU {c.section_title}) costs Rs {price.group(1)}")
        if not lines:
            return ""  # nothing above the threshold: the prompt stays exactly as before
        act = ("call create_ticket right away, then tell the customer a human colleague will contact them; do not wait for an order ID"
               if self.desk else "tell the customer that a human colleague will need to handle it and offer to connect them")
        return (f"ESCALATION CHECK (computed by our system from the catalog and rule ESC-1): {'; '.join(lines)} - above the Rs {threshold} "
                f"high-value threshold of ESC-1 rule 1. If the customer says this product is damaged, defective or the wrong item, hand over to a human now: {act}.")

    def respond(self, text: str, channel: str = "chat") -> AgentResult:
        a, desk = self.a, self.desk
        before = len(desk.events) if desk else 0
        if desk:
            desk.process_user_text(text)

        rq = self._rewrite(text)
        retrieved = a.retriever.search([text, rq], k=a.s.top_k, dense_query=rq) if rq else []
        shown = list(retrieved)
        for c in a.standing:
            if c not in shown:
                shown.append(c)

        blocks = []
        if desk:
            blocks += [f"CALL STATE:\n{desk.state_text()}", f"ORDER FACTS (from our system):\n{desk.facts_text()}"]
        hint = self._value_hint(shown)
        if hint:
            blocks.append(hint)
        blocks += ["CONTEXT (policy passages; use only these for policy answers):\n" + "\n\n".join(c.for_prompt() for c in shown),
                   f"CHANNEL: {'phone call (spoken aloud)' if channel == 'voice' else 'chat'}", f"CUSTOMER MESSAGE:\n{text}"]
        system = prompts.CONVO_SYSTEM if desk else prompts.RAG_SYSTEM
        tools, funcs = a.calendar.schemas(), a.calendar.functions()
        if desk:
            tools, funcs = tools + [TICKET_SCHEMA], {**funcs, "create_ticket": desk.create_ticket}
        messages = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": "\n\n".join(blocks)}]

        resp = a.llm.chat(messages, tools=tools, tool_funcs=funcs)
        reply = resp.text.strip() or FALLBACK
        if desk and not desk.ticket_id and CLAIM.search(reply):
            # Honesty guard: the reply talks about a ticket / handover but our code never created one. Ask the model to fix it.
            fix = messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": (
                "SYSTEM CHECK: no ticket exists yet, so you must not say one was created or that a colleague will contact the customer. "
                "If a handover is really required (an escalation rule applies, or the customer asked for a person), call create_ticket now "
                "and then reply. Otherwise rewrite your reply without claiming a ticket or handover.")}]
            resp2 = a.llm.chat(fix, tools=tools, tool_funcs=funcs)
            if resp2.text.strip():
                reply = resp2.text.strip()
            resp.tool_calls = list(resp.tool_calls) + list(resp2.tool_calls)
        if desk:
            reply = desk.redact(reply) or FALLBACK
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]

        from .orders import ToolEvent
        events = list(desk.events[before:]) if desk else []
        events += [ToolEvent(t.name, t.arguments, t.result if isinstance(t.result, dict) else {"result": t.result}, "llm")
                   for t in resp.tool_calls if t.name != "create_ticket"]
        return AgentResult(reply, [c.chunk_id for c in shown], [c.chunk_id for c in retrieved], rq, events, desk.ticket_id if desk else "")
