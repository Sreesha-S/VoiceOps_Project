"""Turn the knowledge base into searchable chunks.

Chunking choice: ONE CHUNK PER SECTION (`## [RET-2] Return windows ...`).
  * Each section is a short self-contained rule, so a section is the natural unit of meaning.
  * Cutting by a fixed number of characters would slice tables (like the refund-timeline table) in half.
  * We keep the section ID, document type, status and effective date with every chunk: retrieval is
    scored on these IDs, and the precedence rule (policy > guide > help > FAQ > marketing) needs them.
Each row of products.csv also becomes a chunk (id "products.csv:ELE-0012"), because some answers need
two sources: a policy plus a catalog row.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$", re.M)
_FRONT = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


@dataclass
class Chunk:
    chunk_id: str
    doc_type: str      # policy | agent_guide | help | faq | marketing | meta | catalog
    status: str        # current | superseded
    effective_from: str
    doc_title: str
    section_title: str
    text: str
    note: str = ""     # e.g. "this version only applies to electronics delivered before 1 Aug 2026"

    def for_index(self) -> str:
        return "\n".join(p for p in (self.doc_title, self.section_title, self.text, self.note) if p)

    def for_prompt(self) -> str:
        meta = f"{self.doc_type}, {self.status}" + (f", effective {self.effective_from}" if self.effective_from else "")
        body = self.text + (f"\nNote: {self.note}" if self.note else "")
        return f"[{self.chunk_id}] {self.doc_title} > {self.section_title} ({meta})\n{body}"


def _front_matter(raw: str):
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def load_kb_chunks(kb_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
        found = list(_SECTION.finditer(body))
        if not found:
            continue
        preamble = body[: found[0].start()]
        note = "\n".join(l.lstrip("> ").strip() for l in preamble.splitlines() if l.strip() and not l.startswith("# "))
        for i, m in enumerate(found):
            end = found[i + 1].start() if i + 1 < len(found) else len(body)
            chunks.append(Chunk(
                chunk_id=m.group(1).strip(),
                doc_type=meta.get("doc_type", ""),
                status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", ""),
                doc_title=meta.get("title", path.stem),
                section_title=m.group(2).strip(),
                text=body[m.end():end].strip(),
                note=note if meta.get("status") == "superseded" else "",
            ))
    return chunks


def load_product_chunks(products_csv: Path) -> list[Chunk]:
    chunks = []
    with open(products_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text = (f"{r['title']} (SKU {r['sku']}), brand {r['brand']}. return_category: {r['return_category']}. "
                    f"price: Rs {r['price_inr']}. fulfilled_by: {r['fulfilled_by']}. assured: {r['assured']}. "
                    f"brand warranty: {r['warranty_months']} months. {r['description']}")
            chunks.append(Chunk(f"products.csv:{r['sku']}", "catalog", "current", "", "Product catalog", r["sku"], text))
    return chunks


def load_all_chunks(dataset_dir: Path) -> list[Chunk]:
    d = Path(dataset_dir)
    return load_kb_chunks(d / "kb") + load_product_chunks(d / "data" / "products.csv")
