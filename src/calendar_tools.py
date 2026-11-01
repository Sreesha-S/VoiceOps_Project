"""Small exact tools the LLM can call. Language models are unreliable at calendar maths, so code does it.

Business-day rule (KB-00.3): Monday-Saturday, skipping holidays.csv. Counting starts the day AFTER the
start date. Example: refund started Sat 17 Oct 2026, "5 business days" -> Sat 24 Oct
(18 Oct is a Sunday and 19 Oct is a warehouse holiday).
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


def to_date(s) -> date:
    return date.fromisoformat(str(s).strip()[:10])


def pretty(d) -> str:
    d = d if isinstance(d, date) else to_date(d)
    return f"{d.day} {d.strftime('%B %Y')}"


class Calendar:
    def __init__(self, dataset_dir: Path, today: str):
        self.today = to_date(today)
        data = Path(dataset_dir) / "data"
        with open(data / "holidays.csv", encoding="utf-8", newline="") as fh:
            self.holidays = {to_date(r["date"]) for r in csv.DictReader(fh)}
        with open(data / "pincodes.csv", encoding="utf-8", newline="") as fh:
            self.pincodes = {r["pincode"]: r for r in csv.DictReader(fh)}

    def is_business_day(self, d: date) -> bool:
        return d.weekday() != 6 and d not in self.holidays

    def add_business_days(self, start: date, n: int) -> date:
        d, count = start, 0
        while count < n:
            d += timedelta(days=1)
            count += self.is_business_day(d)
        return d

    def business_days_between(self, start: date, end: date) -> int:
        d, count = start, 0
        while d < end:
            d += timedelta(days=1)
            count += self.is_business_day(d)
        return count

    # ---- tools exposed to the LLM
    def date_calc(self, operation: str, date_str: str, n: int = 0, end_date: str = "", min_days: int = 0, max_days: int = 0) -> dict:
        start = to_date(date_str)
        if operation == "business_day_range":  # "N to M business days" in ONE call, so the model never chains two calculations
            lo, hi = self.add_business_days(start, int(min_days)), self.add_business_days(start, int(max_days))
            return {"earliest_iso": lo.isoformat(), "latest_iso": hi.isoformat(), "text": f"between {pretty(lo)} and {pretty(hi)}"}
        if operation == "add_business_days":
            res = self.add_business_days(start, int(n))
        elif operation == "add_calendar_days":
            res = start + timedelta(days=int(n))
        elif operation == "business_days_between":
            return {"business_days": self.business_days_between(start, to_date(end_date))}
        elif operation == "calendar_days_between":
            return {"calendar_days": (to_date(end_date) - start).days}
        else:
            return {"error": f"unknown operation {operation}"}
        return {"result_iso": res.isoformat(), "result_text": pretty(res), "weekday": res.strftime("%A")}

    def check_pincode(self, pincode: str) -> dict:
        row = self.pincodes.get("".join(c for c in str(pincode) if c.isdigit()))
        return {"found": True, **row} if row else {"found": False, "note": "Pincode is not in the serviceable list."}

    def schemas(self) -> list:
        return [
            {"type": "function", "function": {
                "name": "date_calc",
                "description": ("Exact date arithmetic. ALWAYS use it for deadlines and timelines instead of computing in your head. "
                                "For a timeline written as 'N to M business days' use business_day_range ONCE (date_str = the day the refund/shipment started, min_days = N, max_days = M) and quote its text exactly. "
                                "add_business_days skips Sundays and company holidays; counting starts the day AFTER date_str. "
                                "add_calendar_days is for return windows. The *_between operations count days up to end_date."),
                "parameters": {"type": "object", "properties": {
                    "operation": {"type": "string", "enum": ["business_day_range", "add_business_days", "add_calendar_days", "business_days_between", "calendar_days_between"]},
                    "date_str": {"type": "string", "description": "Start date as YYYY-MM-DD"},
                    "n": {"type": "integer", "description": "Number of days to add"},
                    "end_date": {"type": "string", "description": "End date as YYYY-MM-DD (for *_between)"},
                    "min_days": {"type": "integer", "description": "Shortest timeline in business days (business_day_range)"},
                    "max_days": {"type": "integer", "description": "Longest timeline in business days (business_day_range)"}},
                    "required": ["operation", "date_str"]}}},
            {"type": "function", "function": {
                "name": "check_pincode",
                "description": "Is a pincode serviceable, and are cash on delivery and Express available there?",
                "parameters": {"type": "object", "properties": {"pincode": {"type": "string"}}, "required": ["pincode"]}}},
        ]

    def functions(self) -> dict:
        return {"date_calc": self.date_calc, "check_pincode": self.check_pincode}
