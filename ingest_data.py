"""IBM Knowledge RAG Assistant - Data Ingestion
================================================

Pipeline Responsibilities
-------------------------
1. Load configuration (inline defaults + optional overrides via environment variables or files).
2. Collect IBM-hosted PDFs (limited to allowed domains) with retry logic.
3. Scrape a curated set of IBM blog/article pages; clean, deduplicate, and persist unified text.
4. Generate a synthetic HR-style CSV dataset (deterministic for reproducibility).
5. Emit a JSON manifest summarizing ingested artifacts (paths, counts, timestamps).

Run:
        python ingest_data.py

Configuration Overrides
-----------------------
- Custom PDF URL file: ``data/ibm_pdf_urls.txt`` (one IBM PDF URL per line).
- Environment variables (optional):
    * ``INGEST_MAX_PDF=all`` download all PDFs (default if unset).
    * ``INGEST_MAX_PDF=<number>`` limit number of PDFs.
    * ``INGEST_MAX_SCRAPE_PAGES=5`` cap pages visited.
    * ``INGEST_MIN_PARAGRAPHS=15`` minimum paragraphs before fallback text injection.

Outputs
-------
- ``data/pdfs/`` downloaded PDFs
- ``data/website_text.txt`` cleaned aggregated paragraphs
- ``data/ibm_hr.csv`` synthetic HR dataset
- ``data/ingest_manifest.json`` summary of the run

Design Notes
------------
- Resilient networking via a shared ``requests.Session`` with backoff retries.
- Paragraph cleaning removes extra whitespace and deduplicates identical paragraphs.
- Explicit domain filtering ensures IBM-only PDF sources.
- Manifest enables downstream indexing sanity checks and reproducible audits.
"""
from __future__ import annotations

import os
import time
import csv
import json
import random
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
WEBSITE_TEXT_PATH = DATA_DIR / "website_text.txt"
CSV_PATH = DATA_DIR / "ibm_hr.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# IBM PDF SOURCES
# ----------------
# The project now restricts to IBM-hosted PDF links ONLY (no external fallbacks) per user request.
# You can override the list by creating a file: data/ibm_pdf_urls.txt with one IBM PDF URL per line.
# Any non-IBM domain URLs will be ignored (simple domain filter).
# NOTE: Replace the placeholder/example URLs below with the exact IBM publication links you want.
# If a URL 404s, it will simply be skipped (script will still continue with other data sources).

DEFAULT_IBM_PDF_URLS = [
    # IBM Research & AI Whitepapers
    "https://research.ibm.com/downloads/ces_2021/IBMResearch_STO_2021_Whitepaper.pdf",

    # IBM R&D and Innovation Overview
    "https://public.dhe.ibm.com/software/applications/plm/landingdtw/lit_auto_researchdevelopmentinnovating.pdf",

    # IBM Data Quality Whitepaper
    "https://public.dhe.ibm.com/software/data/integration/whitepapers/data_quality.pdf",

    # IBM HMC Connectivity Security White Paper
    "https://public.dhe.ibm.com/linux/service_agent_code/LINUX/HMC_SECURITY_WHITEPAPER.PDF",

    # IBM Training and Value of Workforce Education
    "https://www.ibm.com/training/pdfs/IBMTraining-TheValueofTraining.pdf",

    # IBM Research Brochure (Zurich Lab)
    "https://public.dhe.ibm.com/software/data/persontopersonprogram/cologne/heusler.pdf",

    # IBM TS7700 R6.0 Performance White Paper
    "https://www.ibm.com/support/pages/system/files/inline-files/White%20Paper%20-%20TS7700_R6.0%20Performance_v_1.0.pdf",

    # IBM i2 Analyze Platform Architecture Whitepaper
    "https://www.ibm.com/support/pages/system/files/inline-files/platform_architecture_whitepaper_external_pdf.pdf",

    # IBM Global Research Overview
    "https://www.ibm.com/investor/att/pdf/Research.pdf",

    # IBM + Oracle Database 19c Integration
    "https://www.ibm.com/support/pages/system/files/inline-files/Flash__Oracle_DB_19c_SLES_15_2021.pdf",
]


IBM_DOMAIN = "ibm.com"

BLOG_SEEDS = [
    # IBM Research main blog
    "https://research.ibm.com/blog",

    # IBM Artificial Intelligence main page
    "https://www.ibm.com/ai",

    # IBM Think Blog (AI and technology insights)
    "https://www.ibm.com/blog/tag/ibm-think/",

    # IBM Consulting AI Blog (Newsroom)
    "https://newsroom.ibm.com/blog-how-were-making-ai-sticky-in-ibm-consulting",

    # IBM Research – AI Inference Explained
    "https://research.ibm.com/blog/AI-inference-explained",

    # IBM Research – AI FactSheets (Trustworthy AI)
    "https://research.ibm.com/blog/aifactsheets",

    # IBM Research – Neural Networks (NeuNets)
    "https://research.ibm.com/blog/neunets",

    # IBM Research – DualTKB (Knowledge Graphs)
    "https://research.ibm.com/blog/dualtkb",

    # IBM Think Insights – AI Trends
    "https://www.ibm.com/think/insights/artificial-intelligence-trends",

    # IBM Think Insights – Impact of AI
    "https://www.ibm.com/think/insights/impact-of-ai",

    # IBM Think Topics – AI in the Workplace
    "https://www.ibm.com/think/topics/ai-in-the-workplace",
]


def ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure parent exists for manifest even if no PDFs downloaded
    (DATA_DIR).mkdir(parents=True, exist_ok=True)


def load_ibm_pdf_urls() -> List[str]:
    """Load IBM PDF URLs from optional file or fallback to defaults.

    Will filter to allowed IBM domains only.
    File path: data/ibm_pdf_urls.txt
    """
    urls: List[str] = []
    override_file = DATA_DIR / "ibm_pdf_urls.txt"
    if override_file.exists():
        for line in override_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    if not urls:
        urls = DEFAULT_IBM_PDF_URLS
    # Filter by IBM domain (*.ibm.com) and .pdf extension
    filtered: List[str] = []
    for u in urls:
        try:
            host = (urlparse(u).hostname or "").lower()
            if (host == IBM_DOMAIN or host.endswith(f".{IBM_DOMAIN}")) and u.lower().endswith('.pdf'):
                filtered.append(u)
        except Exception:
            continue
    return filtered


def build_session() -> requests.Session:
    """Create a requests session with basic retry/backoff."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_pdfs(urls: List[str], out_dir: Path, max_files: int | None = None, session: requests.Session | None = None) -> List[Path]:
    """Download IBM PDFs; skip if already exists.

    Set `max_files=None` to download all. Returns list of saved Path objects.
    Gracefully skips failures.
    """
    saved: List[Path] = []
    session = session or build_session()
    for url in urls:
        if max_files is not None and len(saved) >= max_files:
            break
        try:
            fn = url.split("/")[-1]
            if not fn.endswith(".pdf"):
                fn += ".pdf"
            dest = out_dir / fn
            if dest.exists() and dest.stat().st_size > 2048:  # basic size sanity
                saved.append(dest)
                continue
            resp = session.get(url, headers=HEADERS, timeout=25)
            ctype = resp.headers.get("content-type", "").lower()
            if resp.status_code == 200 and "pdf" in ctype:
                dest.write_bytes(resp.content)
                saved.append(dest)
            else:
                continue
        except Exception:
            continue
        time.sleep(0.4)
    return saved


def clean_paragraph(text: str) -> str:
    return " ".join(text.split()).strip()


def scrape_blogs(
    seeds: List[str],
    out_path: Path,
    max_pages: int = 3,
    min_paragraphs: int = 10,
    session: requests.Session | None = None,
) -> int:
    """Scrape seed pages; collect cleaned paragraphs; persist text file.

    Deduplicates paragraphs and injects fallback contextual text if below threshold.
    Returns final paragraph count.
    """
    paragraphs: List[str] = []
    seen: set[str] = set()
    visited = 0
    session = session or build_session()
    for seed in seeds:
        if visited >= max_pages:
            break
        try:
            r = session.get(seed, headers=HEADERS, timeout=25)
            if r.status_code != 200:
                visited += 1
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for p in soup.find_all("p"):
                raw = p.get_text(strip=True)
                if not raw:
                    continue
                cleaned = clean_paragraph(raw)
                if len(cleaned.split()) < 6:
                    continue
                if cleaned in seen:
                    continue
                paragraphs.append(cleaned)
                seen.add(cleaned)
            visited += 1
        except Exception:
            visited += 1
            continue
        time.sleep(0.35)

    if len(paragraphs) < min_paragraphs:
        paragraphs.extend(
            [
                "IBM Research focuses on trustworthy AI, scalable foundation models, and hybrid cloud acceleration.",
                "Enterprise AI adoption at IBM emphasizes governance, security, and responsible deployment.",
                "Workforce analytics enables proactive insights into attrition, engagement, and performance factors.",
            ]
        )

    out_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return len(paragraphs)


def generate_hr_csv(out_path: Path, n: int = 50) -> None:
    """Generate deterministic synthetic HR-style CSV (small size)."""
    fields = [
        "EmployeeNumber",
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "YearsAtCompany",
        "Attrition",
        "JobSatisfaction",
        "PerformanceRating",
    ]
    departments = ["Research & Development", "Sales", "Human Resources"]
    roles = ["Data Scientist", "Research Scientist", "Sales Executive", "HR Analyst"]

    random.seed(42)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(1, n + 1):
            row = [
                i,
                random.randint(21, 60),
                random.choice(departments),
                random.choice(roles),
                random.randint(3000, 20000),
                random.randint(0, 20),
                random.choice(["Yes", "No"]),
                random.randint(1, 4),
                random.randint(1, 4),
            ]
            writer.writerow(row)


def _pick_existing_csv_in_root() -> Path | None:
    """Find a CSV placed in the project root, preferring names with 'ibm' or 'hr'."""
    candidates = list(PROJECT_ROOT.glob("*.csv"))
    if not candidates:
        return None

    def score(p: Path) -> int:
        name = p.name.lower()
        s = 0
        if "ibm" in name:
            s += 2
        if "hr" in name or "human" in name:
            s += 1
        return s

    candidates.sort(key=lambda p: (score(p), p.stat().st_size if p.exists() else 0), reverse=True)
    return candidates[0]


def ensure_hr_csv(csv_target: Path = CSV_PATH) -> tuple[Path, str]:
    """Ensure a CSV exists at data/ibm_hr.csv.

    Order of precedence:
    - HR_CSV_PATH env var points to file -> copy to target (mode='env')
    - CSV detected in project root -> copy to target (mode='copied')
    - Else generate synthetic CSV (mode='generated')
    - If target already exists -> return as 'existing'
    """
    if csv_target.exists() and csv_target.stat().st_size > 0:
        return csv_target, "existing"

    env_path = os.getenv("HR_CSV_PATH")
    if env_path:
        src = Path(env_path)
        if src.exists() and src.is_file():
            csv_target.parent.mkdir(parents=True, exist_ok=True)
            csv_target.write_bytes(src.read_bytes())
            return csv_target, "env"

    found = _pick_existing_csv_in_root()
    if found is not None and found.exists():
        csv_target.parent.mkdir(parents=True, exist_ok=True)
        csv_target.write_bytes(found.read_bytes())
        return csv_target, "copied"

    # Fallback: generate
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    generate_hr_csv(csv_target, n=80)
    return csv_target, "generated"


def write_manifest(data: Dict[str, Any]) -> Path:
    manifest_path = DATA_DIR / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    ensure_dirs()

    # Config via environment overrides
    # INGEST_MAX_PDF: "all"/unset => download all, or integer to limit
    mp_env = os.getenv("INGEST_MAX_PDF", "").strip().lower()
    if mp_env in ("", "all", "unlimited", "none", "inf", "-1", "0"):
        max_pdfs: int | None = None
    else:
        try:
            max_pdfs = max(0, int(mp_env))
        except ValueError:
            max_pdfs = None
    max_pages = int(os.getenv("INGEST_MAX_SCRAPE_PAGES", "5"))
    min_paragraphs = int(os.getenv("INGEST_MIN_PARAGRAPHS", "15"))

    session = build_session()
    ibm_urls = load_ibm_pdf_urls()
    print("[INGEST] Downloading IBM PDFs...")
    pdfs = download_pdfs(ibm_urls, PDF_DIR, max_files=max_pdfs, session=session)
    print(f"[INGEST] PDFs saved: {len(pdfs)} of {len(ibm_urls)} -> {[p.name for p in pdfs]}")

    print("[INGEST] Scraping IBM pages...")
    n_paragraphs = scrape_blogs(
        BLOG_SEEDS,
        WEBSITE_TEXT_PATH,
        max_pages=max_pages,
        min_paragraphs=min_paragraphs,
        session=session,
    )
    print(f"[INGEST] Paragraphs collected: {n_paragraphs} -> {WEBSITE_TEXT_PATH}")

    print("[INGEST] Resolving HR CSV...")
    final_csv, csv_mode = ensure_hr_csv(CSV_PATH)
    print(f"[INGEST] CSV ready: {final_csv} (mode={csv_mode})")

    manifest = {
        "pdf_count": len(pdfs),
        "pdf_files": [p.name for p in pdfs],
        "pdf_source_urls": ibm_urls,
        "pdf_limit": (max_pdfs if max_pdfs is not None else "all"),
        "paragraph_count": n_paragraphs,
        "csv_rows": (sum(1 for _ in open(final_csv, "r", encoding="utf-8", errors="ignore")) - 1) if final_csv.exists() else 0,
        "csv_path": str(final_csv),
        "csv_mode": csv_mode,
        "config": {
            "max_pdfs": (max_pdfs if max_pdfs is not None else "all"),
            "max_pages": max_pages,
            "min_paragraphs": min_paragraphs,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = write_manifest(manifest)
    print(f"[INGEST] Manifest written: {manifest_path}")

    print("[INGEST] Done. Dataset prepared in 'data/' directory.")


if __name__ == "__main__":
    main()
