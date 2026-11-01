"""STEP 3 plumbing: who is the caller, is the caller verified, and what does our system know about the order?

Design rule: CODE (not the LLM) does identity checks. The LLM can be persuaded or can hallucinate; code cannot.
  * Code reads order IDs / verification digits out of what the customer said (spoken numbers included),
    calls the dataset's lookup_order tool, counts failed attempts (max 2) and locks after that.
  * The LLM only ever sees order details AFTER verification succeeded, and never the secrets themselves.
  * Tickets are created through a wrapper that removes secrets from the summary and records whether the
    tool really returned created=True. So the agent can only claim a ticket if one truly exists.
"""
from __future__ import annotations

import importlib
import re
import sys
import unicodedata
from dataclasses import dataclass, field

from .calendar_tools import Calendar, pretty, to_date

DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
               "eight": "8", "nine": "9", "shunya": "0", "ek": "1", "teen": "3", "char": "4", "paanch": "5",
               "panch": "5", "chhe": "6", "saat": "7", "aath": "8", "nau": "9"}
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE = re.compile(rf"\b(\d{{1,2}}\s*(st|nd|rd|th)?\s+)?({MONTHS})\.?,?\s+\d{{1,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", re.I)


# ------------------------------------------------------------------ reading numbers out of speech/text
def ascii_digits(text: str) -> str:
    """'२९४९' (Devanagari) -> '2949'. Also works for other Indian scripts."""
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in text)


def number_runs(text: str) -> list[str]:
    t = _DATE.sub(" ", ascii_digits(text))
    words, out, i = t.split(), [], 0
    while i < len(words):
        j = i
        while j < len(words) and re.sub(r"\W", "", words[j]).lower() in DIGIT_WORDS:
            j += 1
        if j - i >= 4:  # "four five three three" -> 4533 (a single "four" stays a word)
            out.append("".join(DIGIT_WORDS[re.sub(r"\W", "", w).lower()] for w in words[i:j]))
            i = j
        else:
            out.append(words[i])
            i += 1
    return [re.sub(r"\D", "", r) for r in re.findall(r"\d(?:[ \-]?\d)*", " ".join(out))]


def parse_numbers(text: str):
    """Returns (order_id or None, verification_value or None). 8 digits = order, 4 = phone suffix, 6 = pincode."""
    order = ver = None
    for r in number_runs(text):
        if len(r) == 8 and order is None:
            order = r
        elif len(r) in (12, 14) and order is None:
            order, ver = r[:8], r[8:]
        elif len(r) in (4, 6):
            ver = r
    return order, ver


def is_numbers_only(text: str) -> bool:
    letters = re.sub(r"[\d\s\-.,।:]", "", ascii_digits(text))
    return len(letters) < 3


# ------------------------------------------------------------------ tool events
@dataclass
class ToolEvent:
    name: str
    args: dict
    result: dict
    source: str  # "code" = our program called it, "llm" = the model asked for it

    def public(self) -> dict:
        """Safe-to-save version: secrets and private order details removed."""
        args = {k: ("****" if k == "verification_value" else v) for k, v in self.args.items()}
        if self.name == "lookup_order":
            res = {k: self.result.get(k) for k in ("found", "verified", "verification_required") if k in self.result}
        else:
            res = self.result
        return {"name": self.name, "args": args, "result": res, "source": self.source}


def load_mock_tools(dataset_dir):
    p = str(dataset_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("tools")


TICKET_SCHEMA = {"type": "function", "function": {
    "name": "create_ticket",
    "description": ("Hand the case over to a human colleague by creating a ticket. Call it BEFORE telling the customer a human will "
                    "contact them. The summary must contain: the customer's issue, the detected intent, the order ID if any, what the "
                    "customer already told you, what you already checked, and the reason for escalation. Never include verification digits, "
                    "phone numbers, emails or addresses. Returns created=true only if the ticket really exists."),
    "parameters": {"type": "object", "properties": {
        "summary": {"type": "string"}, "reason": {"type": "string", "description": "short code, e.g. human_requested, high_value_damage, repeat_contact"},
        "intent": {"type": "string", "description": "e.g. refund_status, damaged_item"}}, "required": ["summary"]}}}


@dataclass
class OrderDesk:
    tools: object
    calendar: Calendar
    order_id: str = ""
    verified: bool = False
    locked: bool = False
    failed: int = 0
    order: dict = field(default_factory=dict)
    secrets: set = field(default_factory=set)
    ticket_id: str = ""
    events: list = field(default_factory=list)
    MAX_ATTEMPTS = 2

    # ---- called once per customer message, BEFORE the LLM answers
    def process_user_text(self, text: str) -> None:
        order_id, ver = parse_numbers(text)
        if order_id and order_id != self.order_id:  # a new order number: start over
            self.order_id, self.verified, self.locked, self.failed, self.order = order_id, False, False, 0, {}
            res = self.tools.lookup_order(order_id)  # no verification value yet: the tool only asks for it
            self.events.append(ToolEvent("lookup_order", {"order_id": order_id}, res, "code"))
        if ver and self.order_id and not self.verified and not self.locked:
            self.secrets.add(ver)
            res = self.tools.lookup_order(self.order_id, verification_value=ver)
            self.events.append(ToolEvent("lookup_order", {"order_id": self.order_id, "verification_value": ver}, res, "code"))
            if res.get("verified"):
                self.verified, self.order = True, res
            else:
                self.failed += 1
                self.locked = self.failed >= self.MAX_ATTEMPTS

    # ---- what the LLM is told about the call
    def state_text(self) -> str:
        if not self.order_id:
            return "No order ID yet. If the customer needs help with a specific order, ask for the order ID (one question)."
        spaced = f"{self.order_id[:4]} {self.order_id[4:]}"
        if self.verified:
            return f"Order {spaced}: the customer is VERIFIED by our system. You may discuss this order using ORDER FACTS."
        if self.locked:
            return (f"Order {spaced}: verification FAILED twice. Do NOT share any order details. Offer help with general policy questions "
                    "and suggest the customer contact support from the phone number registered on the account.")
        if self.failed == 1:
            return (f"Order {spaced}: the verification value did not match (attempt 1 of 2). Say it did not match, without hinting at the "
                    "correct value, and ask once more for ONE of: last 4 digits of the registered phone number OR the delivery pincode.")
        return (f"Order {spaced}: NOT verified yet. Share no order details. Ask for ONE of: the last 4 digits of the registered phone number "
                "OR the delivery pincode for this order.")

    def facts_text(self) -> str:
        if not self.verified:
            return "(none - customer not verified)"
        o, today = self.order, self.calendar.today
        hide = {"customer_id", "delivery_pincode", "items", "found", "verified", "timeline", "order_id"}
        lines = [f"today: {today.isoformat()} ({pretty(today)})", f"order_id: {o['order_id']}"]
        for k, v in o.items():
            if k in hide or v is None or v == "":
                continue
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
                v = f"{v[:10]} ({pretty(v)})"
            lines.append(f"{k}: {v}")
        if o.get("timeline"):
            lines.append(f"timeline: {o['timeline']}")
        for it in o.get("items", []):
            lines.append(f"item line {it['line_no']}: {it['title']} (sku {it['sku']}, return_category {it['return_category']}, "
                         f"qty {it['qty']}, unit price Rs {it['unit_price_inr']})")
        # facts computed by code (so the model does not have to count days itself)
        if o.get("delivered_on"):
            lines.append(f"calendar days since delivery: {(today - to_date(o['delivered_on'])).days}")
            if o.get("promised_date"):
                late = (to_date(o["delivered_on"]) - to_date(o["promised_date"])).days
                lines.append(f"delivered {late} day(s) after the promised date" if late > 0 else "delivered on or before the promised date")
        elif o.get("promised_date") and o.get("status") not in ("cancelled", "returned_to_origin", "rto"):
            late = (today - to_date(o["promised_date"])).days
            if late > 0:
                lines.append(f"not delivered yet; {late} day(s) past the promised date")
        for p in ("cancellation", "rto", "return"):
            if o.get(f"{p}_refund_initiated_on"):
                n = self.calendar.business_days_between(to_date(o[f"{p}_refund_initiated_on"]), today)
                lines.append(f"business days elapsed since the {p} refund was initiated: {n}")
        if o.get("damage_sku"):
            for it in o.get("items", []):
                if it["sku"] == o["damage_sku"]:
                    lines.append(f"damaged/wrong item unit price: Rs {it['unit_price_inr']}")
        return "\n".join(lines)

    # ---- the ticket tool the LLM may call
    def create_ticket(self, summary: str, reason: str = "", intent: str = "") -> dict:
        clean = summary
        for s in set(self.secrets) | ({self.order["delivery_pincode"]} if self.order.get("delivery_pincode") else set()):
            clean = re.sub(rf"(?<!\d){re.escape(s)}(?!\d)", "[hidden]", clean)
        res = self.tools.create_ticket(summary=clean, order_id=self.order_id if self.verified else None, reason=reason or None, intent=intent or None)
        if res.get("created"):
            self.ticket_id = res.get("ticket_id", "")
        self.events.append(ToolEvent("create_ticket", {"summary": clean, "reason": reason, "intent": intent}, res, "llm"))
        return res

    def redact(self, reply: str) -> str:
        for s in self.secrets:
            reply = re.sub(rf"(?<!\d){re.escape(s)}(?!\d)", "", reply)
        return re.sub(r"[ ]{2,}", " ", reply)
