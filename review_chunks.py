"""
review_chunks.py — Dump all COREP chunks to a text file for review.

Usage:
    .venv/Scripts/python review_chunks.py                                   # custom, all files
    .venv/Scripts/python review_chunks.py --parser docling                  # docling, all files
    .venv/Scripts/python review_chunks.py --parser llm --file "Leverage"    # LLM, one file
    .venv/Scripts/python review_chunks.py --parser llm --refresh-maps       # force map regen
    .venv/Scripts/python review_chunks.py --parser llm --out my_review.txt  # custom output name
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from corep_ingest import parse_docx, parse_docx_docling, parse_docx_llm, COREP_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--parser", choices=["custom", "docling", "llm"], default="custom",
                    help="'custom' (default), 'docling', or 'llm'")
parser.add_argument("--file", default=None, metavar="PATTERN",
                    help="Only process files whose name contains PATTERN (case-insensitive).")
parser.add_argument("--out", default=None,
                    help="Output filename. Defaults to chunks_review_{parser}.txt.")
parser.add_argument("--refresh-maps", action="store_true",
                    help="Force regeneration of LLM document maps (llm parser only).")
args = parser.parse_args()

docx_files = sorted(COREP_DIR.glob("*.docx"))
if args.file:
    docx_files = [f for f in docx_files if args.file.lower() in f.name.lower()]

if not docx_files:
    raise SystemExit(f"No matching files in {COREP_DIR}/")

out_path = Path(args.out) if args.out else Path(f"chunks_review_{args.parser}.txt")

if args.parser == "docling":
    parse_fn = lambda p: parse_docx_docling(p)
elif args.parser == "llm":
    parse_fn = lambda p: parse_docx_llm(p, refresh_maps=args.refresh_maps)
else:
    parse_fn = lambda p: parse_docx(p)

total = 0

with out_path.open("w", encoding="utf-8") as f:
    for docx_path in docx_files:
        print(f"  Parsing [{args.parser}]: {docx_path.name}")
        chunks = parse_fn(docx_path)
        f.write(f"\n{'='*80}\n")
        f.write(f"FILE: {docx_path.name}  ({len(chunks)} chunks)\n")
        f.write(f"{'='*80}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- chunk {i} ({len(chunk['content'])} chars) ---\n")
            f.write(chunk["content"])
            f.write("\n\n")
        total += len(chunks)

print(f"\nWrote {total} chunks from {len(docx_files)} file(s) [{args.parser}] to: {out_path}")
