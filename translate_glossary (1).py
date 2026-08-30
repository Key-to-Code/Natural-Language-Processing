
import sys
import csv
import re
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tam_Taml"   
MAX_INPUT_TOKENS = 200


def read_source(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return Path(path).read_text(encoding="utf-8")


def parse_glossary(text: str) -> list[tuple[str, str]]:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    entries: list[tuple[str, str]] = []
    pat = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9]+(?:\s+[A-Za-z0-9()/&\-]+){0,4})\s*[:\-–—]\s+(?P<def>.+)$")

    junk = re.compile(
        r"(https?://|doi\.org|@|this publication|table of contents|"
        r"REV\.|NISTIR|figure \d|^\d+$|page \d)",
        re.IGNORECASE,
    )

    for ln in lines:
        if junk.search(ln):
            continue
        m = pat.match(ln)
        if not m:
            continue
        term = m.group("term").strip()
        definition = m.group("def").strip()
        if 2 <= len(term) <= 45 and 10 <= len(definition) <= 400:
            entries.append((term, definition))

    seen, out = set(), []
    for t, d in entries:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append((t, d))
    return out


def build_translator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        max_length=256,
        truncation=True,
        no_repeat_ngram_size=3,
        num_beams=2,
    )


def translate(translator, tokenizer, text: str) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text, truncation=True, max_length=MAX_INPUT_TOKENS)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return translator(text)[0]["translation_text"]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python translate_glossary.py <glossary.pdf | glossary.txt>")
        sys.exit(1)

    src = sys.argv[1]
    print(f"[1/4] Reading: {src}")
    raw = read_source(src)

    print("[2/4] Parsing glossary entries ...")
    entries = parse_glossary(raw)
    print(f"      Found {len(entries)} entries")
    if not entries:
        print("      No 'Term: definition' entries found.")
        sys.exit(1)

    print(f"[3/4] Loading SLM: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    translator = build_translator()

    print("[4/4] Translating EN -> TA (Tamil)\n")
    rows = []
    header = f"{'#':>3}  {'ENGLISH TERM':<28}  {'TAMIL (தமிழ்)':<24}"
    print(header)
    print("-" * 78)

    for i, (term, definition) in enumerate(entries, 1):
        term_ta = translate(translator, tokenizer, term)
        def_ta = translate(translator, tokenizer, definition)
        rows.append({
            "term_en": term, "definition_en": definition,
            "term_ta": term_ta, "definition_ta": def_ta,
        })
        print(f"{i:>3}  {term:<28.28}  {term_ta}")
        print(f"     EN: {definition}")
        print(f"     TA: {def_ta}\n")

    out_csv = Path("glossary_ta.csv")
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_md = Path("glossary_ta.md")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Bilingual Glossary (English -> Tamil / தமிழ்)\n\n")
        f.write("| # | English term | தமிழ் | English definition | தமிழ் விளக்கம் |\n")
        f.write("|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | {r['term_en']} | {r['term_ta']} | "
                    f"{r['definition_en']} | {r['definition_ta']} |\n")

    print(f"Saved: {out_csv.resolve()}")
    print(f"Saved: {out_md.resolve()}")


if __name__ == "__main__":
    main()