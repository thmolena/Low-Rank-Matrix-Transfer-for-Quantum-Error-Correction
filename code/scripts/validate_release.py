#!/usr/bin/env python3
"""Validate the public negative-audit release and its scientific contract."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
DATA_HASH = "bdfce36a71f04295ac78fb372d9c2e381801c05e3be119f919750ef59026d072"
ROOT_ALLOWLIST = {
    ".gitignore", "LICENSE", "README.md", "code", "index.html", "main.pdf", "main.tex"
}
FIGURES = {
    "fig_method_audit.pdf",
    "fig_rank_audit.pdf",
    "fig_rate_audit.pdf",
    "fig_runtime_audit.pdf",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    root_names = {path.name for path in REPOSITORY.iterdir() if path.name != ".git"}
    require(root_names == ROOT_ALLOWLIST, f"root allowlist mismatch: {sorted(root_names)}")

    data = CODE / "data" / "source" / "syndromes_dataset.zip"
    require(digest(data) == DATA_HASH, "dataset SHA-256 mismatch")
    package_data = CODE / "src" / "hqml_transfer_qec" / "data" / "source" / data.name
    require(digest(package_data) == DATA_HASH, "wheel dataset SHA-256 mismatch")

    results_path = CODE / "results" / "results.json"
    locked_path = CODE / "results" / "locked_results.json"
    require(results_path.read_bytes() == locked_path.read_bytes(), "locked JSON differs")
    results = json.loads(results_path.read_text())
    require(results["rows"] == 75_598 and results["shots"] == 70_000_000, "data counts changed")
    require(results["test_shots"] == 20_000_000, "test split changed")
    require(results["test_labels_used_in_hierarchy"] == 0, "test-label leakage")
    require(results["hierarchy_is_transductive"] == 1, "transductive scope missing")
    methods = results["aggregate_test"]
    require(methods["hamming_kernel"]["log_loss"] < methods["repaired_hierarchy_blend"]["log_loss"], "negative accuracy finding changed")
    require(methods["hamming_kernel"]["brier"] < methods["repaired_hierarchy_blend"]["brier"], "Brier falsification changed")
    runtime = results["runtime_audit"]
    require(
        all(row["dense_apply_seconds"] < row["hierarchy_apply_seconds"] for row in runtime[:-1]),
        "dense timing no longer wins through 1024 records",
    )
    require(
        runtime[-1]["hierarchy_apply_seconds"] < runtime[-1]["dense_apply_seconds"],
        "recorded 2048-row hierarchy crossover disappeared",
    )
    require(results["rank_frontier"][0]["two_rank_condition_observed"] == 0, "rank-two failure disappeared")
    require(results["rank_frontier"][-1]["full_operator_frobenius_relative_error"] < 1e-4, "rank-eight full audit regressed")

    manuscript = (REPOSITORY / "main.tex").read_text()
    counts = {
        "tables": len(re.findall(r"\\begin\{table\}", manuscript)),
        "figures": len(re.findall(r"\\begin\{figure\}", manuscript)),
        "theorems": len(re.findall(r"\\begin\{theorem\}", manuscript)),
        "propositions": len(re.findall(r"\\begin\{proposition\}", manuscript)),
        "proofs": len(re.findall(r"\\begin\{proof\}", manuscript)),
    }
    require(counts == {"tables": 2, "figures": 2, "theorems": 2, "propositions": 0, "proofs": 2}, f"manuscript structure changed: {counts}")
    require("negative hierarchical timing result remain part of the evidence" in manuscript.lower(), "negative disposition missing")
    require("not a hardware decoder or a hierarchical speedup" in manuscript.lower(), "scope disclaimer missing")
    normalized_manuscript = " ".join(manuscript.split())
    require("simulation record, not quantum-hardware data" in normalized_manuscript, "data-origin disclaimer missing")
    require(manuscript.count("\\includegraphics") == 2, "expected two included figures")

    bib = (CODE / "manuscript_assets" / "references.bib").read_text()
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript):
        cited.update(part.strip() for part in group.split(","))
    require(cited <= bib_keys, f"missing bibliography keys: {sorted(cited - bib_keys)}")
    require(len(cited) >= 1, "no primary source is cited")

    for directory in (
        CODE / "results" / "figures",
        CODE / "src" / "hqml_transfer_qec" / "results" / "figures",
    ):
        names = {path.name for path in directory.glob("*.pdf")}
        require(names == FIGURES, f"figure set mismatch in {directory}: {sorted(names)}")
        for path in directory.glob("*.pdf"):
            require(path.read_bytes().startswith(b"%PDF") and path.stat().st_size > 5_000, f"invalid figure: {path}")

    forbidden = {
        "benchmark.py", "benchmark.json", "benchmark.tex", "summary.json",
        "main_results.tex", "paper_tables.tex", "web_overview.png",
        "audit_claims.py", "make_tables.py", "reference_make_figures.py",
        "reference_reproduce.py",
    }
    found = sorted(path.relative_to(CODE).as_posix() for path in CODE.rglob("*") if path.name in forbidden or path.name == "tables")
    require(not found, f"stale artifacts present: {found}")
    require(not any(path.name in {"__pycache__", ".pytest_cache"} for path in CODE.rglob("*")), "cache directory present")

    pdf = REPOSITORY / "main.pdf"
    require(pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size > 50_000, "main.pdf is missing or invalid")
    pdfinfo = shutil.which("pdfinfo")
    pages = None
    if pdfinfo:
        output = subprocess.run([pdfinfo, str(pdf)], check=True, capture_output=True, text=True).stdout
        match = re.search(r"^Pages:\s+(\d+)", output, flags=re.MULTILINE)
        require(match is not None, "could not parse PDF page count")
        pages = int(match.group(1))
        require(3 <= pages <= 12, f"unexpected PDF page count: {pages}")

    print(json.dumps({"release": "PASS", "counts": counts, "pages": pages, "results_sha256": digest(results_path), "pdf_sha256": digest(pdf)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
