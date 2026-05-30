# Text and identifier helpers used across the full pipeline.


import re
from urllib.parse import urlparse


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "untitled"


def url_to_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        path = "root"
    if parsed.query:
        path = f"{path}_{parsed.query}"
    return slugify(path)


from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_snapshot_id() -> str:
    return utc_now().strftime("%Y-%m-%dT%H%M%SZ")


from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    """Return the project root regardless of whether code runs from the repo or notebooks."""
    candidate = (start or Path.cwd()).resolve()
    if candidate.name == "notebooks":
        return candidate.parent
    return candidate


def build_paths(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    vector_store_root = data_root / "vector_store"
    return {
        "root": root,
        "config": root / "config",
        "notebooks": root / "notebooks",
        "src": root / "src",
        "data": data_root,
        "raw": data_root / "raw",
        "normalized": data_root / "normalized",
        "chunks": data_root / "chunks",
        "embeddings": data_root / "embeddings",
        "vector_store": vector_store_root,
        "chroma": vector_store_root / "chroma",
        "manifests": data_root / "manifests",
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key == "root":
            continue
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)


import json
from pathlib import Path
from typing import Iterable


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


import hashlib


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def collect_environment_report(packages: list[str]) -> dict[str, object]:
    package_versions: dict[str, str] = {}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not installed"

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
    }

# Snapshot and artifact saving helpers keep the pipeline reproducible.


import json
from pathlib import Path
from urllib.parse import urlparse



def infer_extension(content_type: str, url: str) -> str:
    lowered = content_type.lower()
    if "html" in lowered:
        return ".html"
    if "json" in lowered:
        return ".json"
    if "pdf" in lowered:
        return ".pdf"

    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else ".bin"


def save_raw_snapshot(
    raw_root: Path,
    source_id: str,
    snapshot_id: str,
    fetch_result: dict,
    item_id: str | None = None,
) -> dict:
    safe_item_id = slugify(item_id) if item_id else url_to_slug(fetch_result["final_url"])
    snapshot_dir = raw_root / source_id / snapshot_id / safe_item_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    extension = infer_extension(fetch_result["content_type"], fetch_result["final_url"])
    body_path = snapshot_dir / f"body{extension}"
    body_bytes = fetch_result["body_bytes"]
    body_path.write_bytes(body_bytes)

    metadata = {
        "snapshot_id": snapshot_id,
        "source_id": source_id,
        "item_id": safe_item_id,
        "url": fetch_result["final_url"],
        "requested_url": fetch_result["requested_url"],
        "fetched_at": fetch_result["fetched_at"],
        "status_code": fetch_result["status_code"],
        "content_type": fetch_result["content_type"],
        "encoding": fetch_result["encoding"],
        "sha256": sha256_bytes(body_bytes),
        "byte_size": len(body_bytes),
        "local_body_path": str(body_path),
        "headers": fetch_result["headers"],
    }

    metadata_path = snapshot_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metadata["metadata_path"] = str(metadata_path)
    return metadata


def save_normalized_documents(normalized_root: Path, snapshot_id: str, source_id: str, documents: list[dict]) -> Path:
    snapshot_dir = normalized_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_path = snapshot_dir / f"{slugify(source_id)}.jsonl"
    write_jsonl(target_path, documents, mode="w")
    return target_path


def save_chunk_records(chunks_root: Path, snapshot_id: str, chunks: list[dict]) -> Path:
    snapshot_dir = chunks_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_path = snapshot_dir / "chunks.jsonl"
    write_jsonl(target_path, chunks, mode="w")
    return target_path

# Fetch helpers enforce the source allowlist and keep the live collection step controlled.


from urllib.parse import urlparse

import requests


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf;q=0.8,*/*;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

MINIMAL_HEADERS = {
    "User-Agent": "rag-k8s-security/0.1 (+assignment ingestion pipeline)",
}

HEADER_PROFILES = (BROWSER_HEADERS, MINIMAL_HEADERS)


def validate_allowed_domain(url: str, allowed_domains: list[str]) -> None:
    host = (urlparse(url).hostname or "").lower()
    if not any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains):
        raise ValueError(f"URL host '{host}' is not in the allowed domain list: {allowed_domains}")


def fetch_url(url: str, allowed_domains: list[str], timeout: int = 30) -> dict:
    validate_allowed_domain(url, allowed_domains)

    last_error: requests.HTTPError | None = None
    response: requests.Response | None = None

    for headers in HEADER_PROFILES:
        response = requests.get(url, timeout=timeout, headers=headers)
        try:
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_error = exc
            if response.status_code != 403:
                raise
    else:
        assert last_error is not None
        raise last_error

    assert response is not None

    return {
        "requested_url": url,
        "final_url": response.url,
        "fetched_at": utc_now_iso(),
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "content_type": response.headers.get("Content-Type", "application/octet-stream"),
        "encoding": response.encoding,
        "body_bytes": response.content,
    }

# HTML normalization keeps section structure when the source is a web page.


import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag


HEADING_TAGS = {"h1", "h2", "h3", "h4"}
TEXT_TAGS = {"p", "li", "pre", "code"}
DROP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "form"}
META_UPDATED_KEYS = (
    "article:modified_time",
    "og:updated_time",
    "last-modified",
    "date",
    "dc.date.modified",
)
PILCROW_RE = re.compile(r"\s*¶\s*")
TRAILING_DASH_RE = re.compile(r"\s*--+\s*$")
LOW_SIGNAL_HEADING_RE = re.compile(
    r"\b(feedback|page status|what'?s next|references|contributors|survey)\b",
    re.IGNORECASE,
)
LOW_SIGNAL_TEXT_RE = re.compile(
    r"(was this page helpful\?|thanks for the feedback|open an issue in the github repository|ask it on stack overflow|"
    r"have feedback for us|changes since first version|learning from first version|raw data and feedback of the survey)",
    re.IGNORECASE,
)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_html_text(text: str) -> str:
    cleaned = PILCROW_RE.sub(" ", text)
    cleaned = TRAILING_DASH_RE.sub("", cleaned)
    cleaned = collapse_whitespace(cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned


def dedupe_consecutive_items(items: list[str]) -> list[str]:
    deduped: list[str] = []
    previous_normalized: str | None = None

    for item in items:
        normalized = clean_html_text(item)
        if not normalized:
            continue
        if normalized == previous_normalized:
            continue
        deduped.append(normalized)
        previous_normalized = normalized

    return deduped


def is_low_signal_section(section_path: list[str], text: str) -> bool:
    joined_headings = " > ".join(section_path)
    if LOW_SIGNAL_HEADING_RE.search(joined_headings):
        return True
    if LOW_SIGNAL_TEXT_RE.search(text):
        return True
    return False


def extract_updated_at(soup: BeautifulSoup) -> str | None:
    for key in META_UPDATED_KEYS:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return collapse_whitespace(tag["content"])

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        return collapse_whitespace(time_tag["datetime"])
    return None


def iter_content_nodes(root: Tag) -> Iterable[Tag]:
    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name in HEADING_TAGS or node.name in TEXT_TAGS:
            yield node


def parse_html_sections(
    html_text: str,
    source_id: str,
    source_url: str,
    trust_level: str,
    topic_tags: list[str],
    document_key: str | None = None,
) -> list[dict]:
    soup = BeautifulSoup(html_text, "lxml")

    for tag_name in DROP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    page_title = clean_html_text((soup.title.string if soup.title and soup.title.string else "Untitled Page"))
    top_heading = root.find("h1")
    document_title = clean_html_text(top_heading.get_text(" ", strip=True)) if top_heading else page_title
    updated_at = extract_updated_at(soup)
    page_key = slugify(document_key) if document_key else url_to_slug(source_url)

    documents: list[dict] = []
    headings: dict[int, str] = {}
    buffer: list[str] = []
    section_counter = 0

    def current_section_path() -> list[str]:
        ordered = [headings[level] for level in sorted(headings)]
        return ordered or [document_title]

    def flush_section() -> None:
        nonlocal section_counter, buffer
        text = clean_html_text(" ".join(dedupe_consecutive_items(buffer)))
        if len(text) < 40:
            buffer = []
            return

        section_counter += 1
        section_path = current_section_path()
        if is_low_signal_section(section_path, text):
            buffer = []
            return
        slug = "__".join(slugify(part) for part in section_path)
        documents.append(
            {
                "doc_id": f"{slugify(source_id)}__{page_key}__{slug}__{section_counter:04d}",
                "snapshot_id": None,
                "source_id": source_id,
                "page_key": page_key,
                "doc_kind": "guide_section",
                "external_id": None,
                "title": document_title,
                "section_path": section_path,
                "source_url": source_url,
                "published_at": None,
                "updated_at": updated_at,
                "page_number": None,
                "trust_level": trust_level,
                "topic_tags": topic_tags,
                "text": text,
            }
        )
        buffer = []

    for node in iter_content_nodes(root):
        node_text = clean_html_text(node.get_text(" ", strip=True))
        if not node_text:
            continue

        if node.name in HEADING_TAGS:
            flush_section()
            level = int(node.name[1])
            headings[level] = node_text
            for deeper_level in list(headings):
                if deeper_level > level:
                    headings.pop(deeper_level, None)
            continue

        if node.name in TEXT_TAGS:
            buffer.append(node_text)

    flush_section()
    return documents


# PDF normalization removes boilerplate and keeps page-level text where useful.


import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from pypdf import PdfReader



NSA_HEADER_RE = re.compile(r"u/oo/168286|pp-22-0324|august 2022 ver\.?|version 1\.?2", re.IGNORECASE)
AGENCY_RE = re.compile(
    r"national security agency|cybersecurity and infrastructure security agency|kubernetes hardening guid",
    re.IGNORECASE,
)
DOT_LEADER_RE = re.compile(r"\.{3,}")
PAGE_ONLY_RE = re.compile(r"^[ivxlcdm0-9]+$", re.IGNORECASE)
PUNCT_ONLY_RE = re.compile(r"^[^\w]+$")
TRAILING_DASH_RE = re.compile(r"\s*--+\s*$")
INDEX_REF_RE = re.compile(r",\s*\d{1,3}(?:-\d{1,3})?\b")
REFERENCES_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
COLLOPHON_RE = re.compile(r"\b(about the author|colophon|acknowledg?ments?)\b", re.IGNORECASE)
ABBREVIATIONS_RE = re.compile(r"\b(symbols and abbreviations|abbreviations)\b", re.IGNORECASE)
THESIS_FRONT_MATTER_RE = re.compile(
    r"\b(master'?s thesis|licensed under a creative commons|preface)\b",
    re.IGNORECASE,
)
BOOK_BACK_MATTER_RE = re.compile(
    r"\b(other books you may enjoy|leave a review|about the reviewers|contributors|about the authors|"
    r"have feedback for us|changes since first version|learning from first version|raw data and feedback of the survey)\b",
    re.IGNORECASE,
)
PACKT_FRONT_MATTER_RE = re.compile(
    r"\b(packt\.com subscribe|first published:|published by packt publishing|foreword)\b",
    re.IGNORECASE,
)
REDHAT_MARKETING_RE = re.compile(
    r"\b(about red hat|redhat\.com|red hat openshift provides a full featured technology stack)\b",
    re.IGNORECASE,
)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_pdf_text(text: str) -> str:
    cleaned = text.replace("\x17", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", cleaned)
    cleaned = collapse_whitespace(cleaned)
    cleaned = TRAILING_DASH_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned


def infer_pdf_title(reader: PdfReader, source_url: str) -> str:
    metadata = reader.metadata or {}
    raw_title = metadata.get("/Title") if hasattr(metadata, "get") else None
    if raw_title:
        return clean_pdf_text(str(raw_title))

    file_name = Path(urlparse(source_url).path).name or "Untitled PDF"
    return clean_pdf_text(file_name)


def normalize_pdf_line(line: str) -> str:
    return clean_pdf_text(line.replace("\x00", " "))


def extract_page_lines(raw_page_text: str) -> list[str]:
    return [normalize_pdf_line(line) for line in raw_page_text.splitlines() if normalize_pdf_line(line)]


def build_repeated_line_counts(page_lines: list[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lines in page_lines:
        counts.update(set(lines))
    return counts


def is_pdf_boilerplate_line(line: str, repeated_counts: Counter[str], total_pages: int) -> bool:
    lowered = line.lower()
    if PAGE_ONLY_RE.fullmatch(line) or PUNCT_ONLY_RE.fullmatch(line):
        return True
    if DOT_LEADER_RE.search(line):
        return True
    if NSA_HEADER_RE.search(lowered):
        return True
    if AGENCY_RE.search(lowered) and repeated_counts.get(line, 0) >= max(3, total_pages // 12):
        return True
    if 0 < len(line) <= 14 and repeated_counts.get(line, 0) >= max(3, total_pages // 10):
        return True
    if repeated_counts.get(line, 0) >= max(5, total_pages // 8) and len(line) <= 120:
        return True
    return False


def is_table_of_contents_page(lines: list[str]) -> bool:
    if not lines:
        return False

    first_lines = " ".join(lines[:6]).lower()
    dot_lines = sum(1 for line in lines if DOT_LEADER_RE.search(line))
    short_lines = sum(1 for line in lines if len(line.split()) <= 10)

    if "contents" in first_lines and (dot_lines >= 2 or short_lines >= 8):
        return True
    return False


def is_index_page(lines: list[str], text: str) -> bool:
    lowered = text.lower()
    if "index |" not in lowered and not lowered.startswith("index "):
        return False

    page_ref_count = len(INDEX_REF_RE.findall(text))
    has_cross_refs = "see also" in lowered
    return page_ref_count >= 8 or has_cross_refs


def is_references_page(lines: list[str], text: str) -> bool:
    lowered = text.lower()
    if not REFERENCES_RE.search(" ".join(lines[:4])):
        return False

    url_count = text.lower().count("http://") + text.lower().count("https://")
    doi_count = lowered.count("doi")
    bracket_count = lowered.count("[") + lowered.count("]")
    return url_count >= 2 or doi_count >= 2 or bracket_count >= 6


def is_abbreviations_page(lines: list[str], text: str) -> bool:
    lowered = text.lower()
    if not ABBREVIATIONS_RE.search(" ".join(lines[:4])):
        return False

    acronym_like = sum(1 for token in text.split() if token.isupper() and 2 <= len(token) <= 8)
    return acronym_like >= 12


def is_packt_front_or_back_matter(page_number: int, text: str) -> bool:
    lowered = text.lower()
    if BOOK_BACK_MATTER_RE.search(lowered):
        return True
    if page_number <= 12 and PACKT_FRONT_MATTER_RE.search(lowered):
        return True
    if page_number <= 10 and "copyright ©" in lowered and "packt publishing" in lowered:
        return True
    if page_number <= 10 and "for my lovely wife" in lowered:
        return True
    return False


def is_redhat_back_matter(page_number: int, text: str) -> bool:
    lowered = text.lower()
    if page_number >= 16 and "copyright ©" in lowered:
        return True
    if page_number >= 15 and REDHAT_MARKETING_RE.search(lowered):
        return True
    return False


def is_cover_or_front_matter_page(page_number: int, lines: list[str], text: str) -> bool:
    lowered = text.lower()
    if page_number == 1 and "cybersecurity technical report" in lowered and "kubernetes hardening guid" in lowered:
        return True
    if is_table_of_contents_page(lines):
        return True
    if is_index_page(lines, text):
        return True
    if is_references_page(lines, text):
        return True
    if COLLOPHON_RE.search(lowered):
        return True
    if is_abbreviations_page(lines, text):
        return True
    if is_packt_front_or_back_matter(page_number, text):
        return True
    if is_redhat_back_matter(page_number, text):
        return True
    if page_number <= 8:
        if lowered.startswith("notices and history") or "document change history" in lowered:
            return True
        if lowered.startswith("publication information"):
            return True
        if lowered.startswith("figures ") or lowered.startswith("figures figure"):
            return True
        if THESIS_FRONT_MATTER_RE.search(lowered):
            return True
    if lowered.startswith("preface ") or lowered == "preface":
        return True
    return False


def parse_pdf_pages(
    pdf_bytes: bytes,
    source_id: str,
    source_url: str,
    trust_level: str,
    topic_tags: list[str],
    document_key: str | None = None,
    min_chars: int = 80,
) -> list[dict]:
    reader = PdfReader(BytesIO(pdf_bytes))
    document_title = infer_pdf_title(reader, source_url)
    page_key = slugify(document_key) if document_key else url_to_slug(source_url)

    raw_page_texts = [page.extract_text() or "" for page in reader.pages]
    page_lines = [extract_page_lines(raw_text) for raw_text in raw_page_texts]
    repeated_counts = build_repeated_line_counts(page_lines)
    total_pages = len(reader.pages)

    documents: list[dict] = []
    for page_number, lines in enumerate(page_lines, start=1):
        filtered_lines = [
            line
            for line in lines
            if not is_pdf_boilerplate_line(line, repeated_counts=repeated_counts, total_pages=total_pages)
        ]
        text = clean_pdf_text(" ".join(filtered_lines))
        if len(text) < min_chars:
            continue
        if is_cover_or_front_matter_page(page_number, filtered_lines, text):
            continue

        documents.append(
            {
                "doc_id": f"{slugify(source_id)}__{page_key}__page_{page_number:04d}",
                "snapshot_id": None,
                "source_id": source_id,
                "page_key": page_key,
                "doc_kind": "pdf_page",
                "external_id": None,
                "title": document_title,
                "section_path": [document_title, f"Page {page_number}"],
                "source_url": source_url,
                "published_at": None,
                "updated_at": None,
                "page_number": page_number,
                "trust_level": trust_level,
                "topic_tags": topic_tags,
                "text": text,
            }
        )

    return documents


# JSON feed normalization converts Kubernetes CVE items into the same schema.


import json
import re



def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_kubernetes_cve_feed(
    feed_bytes: bytes,
    source_id: str,
    source_url: str,
    trust_level: str,
    topic_tags: list[str],
    document_key: str | None = None,
) -> list[dict]:
    payload = json.loads(feed_bytes.decode("utf-8"))

    feed_meta = payload.get("_kubernetes_io", {})
    feed_updated_at = feed_meta.get("updated_at")
    feed_title = collapse_whitespace(
        payload.get("title")
        or payload.get("description")
        or "Kubernetes Official CVE Feed"
    )
    page_key = slugify(document_key) if document_key else url_to_slug(source_url)

    documents: list[dict] = []
    for item in payload.get("items", []):
        item_id = collapse_whitespace(item.get("id") or "unknown_cve")
        summary = collapse_whitespace(item.get("summary") or "")
        content_text = collapse_whitespace(item.get("content_text") or "")
        status = collapse_whitespace(item.get("status") or "")
        published_at = item.get("date_published")
        issue_url = item.get("url")
        advisory_url = item.get("external_url")
        source_link = advisory_url or issue_url or source_url

        text_parts = [
            f"CVE ID: {item_id}",
            f"Summary: {summary}" if summary else "",
            f"Status: {status}" if status else "",
            content_text,
        ]
        text = collapse_whitespace(" ".join(part for part in text_parts if part))
        if len(text) < 40:
            continue

        section_path = [item_id]
        if summary:
            section_path.append(summary)

        documents.append(
            {
                "doc_id": f"{slugify(source_id)}__{page_key}__{slugify(item_id)}",
                "snapshot_id": None,
                "source_id": source_id,
                "page_key": page_key,
                "doc_kind": "cve_feed_item",
                "external_id": item_id,
                "title": feed_title,
                "section_path": section_path,
                "source_url": source_link,
                "published_at": published_at,
                "updated_at": feed_updated_at,
                "page_number": None,
                "trust_level": trust_level,
                "topic_tags": topic_tags,
                "text": text,
            }
        )

    return documents

# Chunking helpers create sentence-aware retrieval units from normalized documents.


import re


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[;:])\s+(?=[A-Z0-9(\"'])")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
LOW_SIGNAL_SECTION_RE = re.compile(
    r"\b(references|what'?s next|what s next|table of contents|contents|contributors|survey)\b",
    re.IGNORECASE,
)
NSA_HEADER_RE = re.compile(r"u/oo/168286|pp-22-0324|august 2022 ver", re.IGNORECASE)
INDEX_LIKE_RE = re.compile(r"\bindex\b|\bsee also\b", re.IGNORECASE)
PAGE_REF_RE = re.compile(r",\s*\d{1,3}(?:-\d{1,3})?\b")
BACK_MATTER_RE = re.compile(
    r"\b(about the author|colophon|acknowledg?ments?|have feedback for us|changes since first version|"
    r"learning from first version|raw data and feedback of the survey)\b",
    re.IGNORECASE,
)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return []

    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(cleaned) if segment.strip()]
    return sentences if sentences else [cleaned]


def split_into_clauses(text: str) -> list[str]:
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return []

    clauses = [segment.strip() for segment in CLAUSE_SPLIT_RE.split(cleaned) if segment.strip()]
    return clauses if clauses else [cleaned]


def build_word_windows(total_words: int, max_words: int, overlap_words: int, min_words: int) -> list[tuple[int, int]]:
    if total_words <= 0:
        return []

    step = max_words - overlap_words
    windows: list[tuple[int, int]] = []
    start = 0

    while start < total_words:
        end = min(start + max_words, total_words)
        current_size = end - start

        if windows and current_size < min_words:
            previous_start, _ = windows[-1]
            windows[-1] = (previous_start, total_words)
            break

        windows.append((start, end))
        if end >= total_words:
            break
        start += step

    return windows


def build_context_prefix(document: dict) -> str:
    parts: list[str] = []
    seen: set[str] = set()

    for raw_value in document.get("section_path", []):
        value = collapse_whitespace(str(raw_value))
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        parts.append(value)

    if not parts:
        return ""

    tail_parts = parts[-2:] if len(parts) >= 2 else parts
    return f"Section: {' > '.join(tail_parts)}. "


def is_url_heavy(text: str) -> bool:
    url_matches = len(URL_RE.findall(text))
    word_count = len(text.split())
    return url_matches >= 2 and (url_matches >= 5 or word_count <= 80)


def is_index_like(text: str) -> bool:
    lowered = text.lower()
    page_ref_count = len(PAGE_REF_RE.findall(text))
    return page_ref_count >= 8 and ("index" in lowered or "see also" in lowered)


def is_low_signal_document(document: dict) -> bool:
    section_text = " > ".join(str(part) for part in document.get("section_path", []))
    text = collapse_whitespace(document.get("text") or "")
    lowered_text = text.lower()

    if LOW_SIGNAL_SECTION_RE.search(section_text):
        return True
    if len(text.split()) <= 15 and NSA_HEADER_RE.search(lowered_text):
        return True
    if is_url_heavy(text):
        return True
    if is_index_like(text):
        return True
    if BACK_MATTER_RE.search(lowered_text):
        return True

    return False


def is_low_signal_chunk(text: str) -> bool:
    cleaned = collapse_whitespace(text)
    word_count = len(cleaned.split())

    if word_count <= 15 and NSA_HEADER_RE.search(cleaned):
        return True
    if is_url_heavy(cleaned):
        return True
    if is_index_like(cleaned):
        return True
    if BACK_MATTER_RE.search(cleaned.lower()):
        return True

    return False


def maybe_enrich_short_text(document: dict, text: str, threshold_words: int = 45) -> str:
    cleaned = collapse_whitespace(text)
    if len(cleaned.split()) >= threshold_words:
        return cleaned

    prefix = build_context_prefix(document)
    if not prefix:
        return cleaned

    if cleaned.lower().startswith(prefix.lower().replace("section: ", "").strip()):
        return cleaned

    return f"{prefix}{cleaned}"


def split_long_fragment(fragment: str, max_words: int, overlap_words: int, min_words: int, strategy: str) -> list[dict]:
    words = fragment.split()
    windows = build_word_windows(
        total_words=len(words),
        max_words=max_words,
        overlap_words=overlap_words,
        min_words=min_words,
    )

    parts: list[dict] = []
    for start, end in windows:
        part_text = " ".join(words[start:end]).strip()
        if not part_text:
            continue
        parts.append(
            {
                "text": part_text,
                "word_count": len(part_text.split()),
                "strategy": strategy,
            }
        )

    return parts


def build_semantic_units(text: str, max_words: int, overlap_words: int, min_words: int) -> list[dict]:
    units: list[dict] = []

    for sentence in split_into_sentences(text):
        sentence_text = collapse_whitespace(sentence)
        if not sentence_text:
            continue

        sentence_word_count = len(sentence_text.split())
        if sentence_word_count <= max_words:
            units.append(
                {
                    "text": sentence_text,
                    "word_count": sentence_word_count,
                    "strategy": "sentence",
                }
            )
            continue

        clause_fragments = split_into_clauses(sentence_text)
        if len(clause_fragments) > 1:
            for clause in clause_fragments:
                clause_text = collapse_whitespace(clause)
                if not clause_text:
                    continue
                clause_word_count = len(clause_text.split())
                if clause_word_count <= max_words:
                    units.append(
                        {
                            "text": clause_text,
                            "word_count": clause_word_count,
                            "strategy": "clause",
                        }
                    )
                else:
                    units.extend(
                        split_long_fragment(
                            fragment=clause_text,
                            max_words=max_words,
                            overlap_words=overlap_words,
                            min_words=min_words,
                            strategy="word_fallback",
                        )
                    )
            continue

        units.extend(
            split_long_fragment(
                fragment=sentence_text,
                max_words=max_words,
                overlap_words=overlap_words,
                min_words=min_words,
                strategy="word_fallback",
            )
        )

    return units


def build_semantic_chunks(units: list[dict], max_words: int, min_words: int) -> list[dict]:
    if not units:
        return []

    target_words = max(min_words, int(max_words * 0.78))
    soft_max_words = max_words + max(12, int(max_words * 0.12))

    ranges: list[tuple[int, int]] = []
    start_index = 0

    while start_index < len(units):
        end_index = start_index
        current_words = 0

        while end_index < len(units):
            next_words = units[end_index]["word_count"]

            if current_words == 0:
                current_words += next_words
                end_index += 1
                continue

            if current_words < min_words:
                current_words += next_words
                end_index += 1
                continue

            if current_words < target_words and current_words + next_words <= soft_max_words:
                current_words += next_words
                end_index += 1
                continue

            if current_words + next_words <= max_words and next_words <= max(18, int(max_words * 0.15)):
                current_words += next_words
                end_index += 1
                continue

            break

        ranges.append((start_index, end_index))

        if end_index >= len(units):
            break

        start_index = end_index

    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        last_words = sum(units[index]["word_count"] for index in range(last_start, last_end))
        if last_words < min_words:
            previous_start, _ = ranges[-2]
            ranges[-2] = (previous_start, last_end)
            ranges.pop()

    chunks: list[dict] = []
    for start_index, end_index in ranges:
        chunk_units = units[start_index:end_index]
        chunk_text = " ".join(unit["text"] for unit in chunk_units).strip()
        if not chunk_text:
            continue

        chunks.append(
            {
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "unit_start": start_index,
                "unit_end": end_index,
                "sentence_count": len(chunk_units),
                "strategies": sorted({unit["strategy"] for unit in chunk_units}),
            }
        )

    return chunks


def build_chunks_from_documents(
    documents: list[dict],
    max_words: int = 160,
    overlap_words: int = 40,
    min_words: int = 40,
) -> list[dict]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")
    if min_words < 1:
        raise ValueError("min_words must be at least 1")

    chunks: list[dict] = []

    for document in documents:
        text = collapse_whitespace(document.get("text") or "")
        if not text or is_low_signal_document(document):
            continue

        units = build_semantic_units(
            text=text,
            max_words=max_words,
            overlap_words=overlap_words,
            min_words=min_words,
        )
        document_chunks = build_semantic_chunks(
            units=units,
            max_words=max_words,
            min_words=min_words,
        )

        source_doc_word_count = len(text.split())

        for chunk_index, chunk in enumerate(document_chunks, start=1):
            enriched_text = maybe_enrich_short_text(document, chunk["text"])
            if is_low_signal_chunk(enriched_text):
                continue

            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}__chunk_{chunk_index:04d}",
                    "snapshot_id": document.get("snapshot_id"),
                    "source_id": document.get("source_id"),
                    "doc_id": document.get("doc_id"),
                    "page_key": document.get("page_key"),
                    "doc_kind": document.get("doc_kind"),
                    "external_id": document.get("external_id"),
                    "title": document.get("title"),
                    "section_path": list(document.get("section_path", [])),
                    "source_url": document.get("source_url"),
                    "published_at": document.get("published_at"),
                    "updated_at": document.get("updated_at"),
                    "page_number": document.get("page_number"),
                    "trust_level": document.get("trust_level"),
                    "topic_tags": list(document.get("topic_tags", [])),
                    "chunk_index": chunk_index,
                    "chunk_unit_start": chunk["unit_start"],
                    "chunk_unit_end": chunk["unit_end"],
                    "sentence_count": chunk["sentence_count"],
                    "chunk_strategies": chunk["strategies"],
                    "source_doc_word_count": source_doc_word_count,
                    "word_count": len(enriched_text.split()),
                    "char_count": len(enriched_text),
                    "text": enriched_text,
                }
            )

    return chunks

# Embedding helpers build retrieval text, dense vectors, and the Chroma store.


import json
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer



def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split()).strip()


def join_path(parts: list[str] | None) -> str:
    if not parts:
        return ""
    return " > ".join(collapse_whitespace(part) for part in parts if collapse_whitespace(part))


def build_retrieval_text(chunk: dict) -> str:
    title = collapse_whitespace(chunk.get("title") or "")
    section_path_text = join_path(chunk.get("section_path"))
    content = collapse_whitespace(chunk.get("text") or "")

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if section_path_text:
        parts.append(f"Section: {section_path_text}")
    if content:
        parts.append(f"Content: {content}")
    return "\n".join(parts)


def build_embedding_records(chunks: list[dict]) -> list[dict]:
    records: list[dict] = []

    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_id": chunk["source_id"],
                "page_key": chunk.get("page_key"),
                "page_number": chunk.get("page_number"),
                "chunk_index": chunk.get("chunk_index"),
                "title": chunk.get("title"),
                "section_path": chunk.get("section_path", []),
                "section_path_text": join_path(chunk.get("section_path")),
                "source_url": chunk.get("source_url"),
                "trust_level": chunk.get("trust_level"),
                "topic_tags": chunk.get("topic_tags", []),
                "topic_tags_text": ", ".join(chunk.get("topic_tags", [])),
                "word_count": chunk.get("word_count"),
                "char_count": chunk.get("char_count"),
                "text": chunk.get("text", ""),
                "retrieval_text": build_retrieval_text(chunk),
            }
        )

    return records


def generate_embeddings(
    records: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> tuple[np.ndarray, dict]:
    model = SentenceTransformer(model_name)
    texts = [record["retrieval_text"] for record in records]

    if texts:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)

    summary = {
        "model_name": model_name,
        "batch_size": batch_size,
        "num_vectors": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] else 0,
    }
    return vectors, summary


def save_embedding_artifacts(
    embeddings_root: Path,
    snapshot_id: str,
    records: list[dict],
    vectors: np.ndarray,
    summary: dict,
) -> dict:
    snapshot_dir = embeddings_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = snapshot_dir / "embeddings.npy"
    metadata_path = snapshot_dir / "metadata.jsonl"
    summary_path = snapshot_dir / "summary.json"

    np.save(vectors_path, vectors)
    write_jsonl(metadata_path, records, mode="w")
    summary_path.write_text(json.dumps({"snapshot_id": snapshot_id, **summary}, indent=2), encoding="utf-8")

    return {
        "embeddings_dir": str(snapshot_dir),
        "vectors_path": str(vectors_path),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
    }


def build_chroma_store(
    chroma_root: Path,
    snapshot_id: str,
    records: list[dict],
    vectors: np.ndarray,
    collection_name: str,
    batch_size: int = 128,
) -> dict:
    snapshot_dir = chroma_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(snapshot_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"snapshot_id": snapshot_id, "distance": "cosine"},
    )

    for start in range(0, len(records), batch_size):
        end = start + batch_size
        batch_records = records[start:end]
        batch_vectors = vectors[start:end]

        collection.add(
            ids=[record["chunk_id"] for record in batch_records],
            embeddings=batch_vectors.tolist(),
            documents=[record["text"] for record in batch_records],
            metadatas=[build_chroma_metadata(record) for record in batch_records],
        )

    return {
        "chroma_dir": str(snapshot_dir),
        "collection_name": collection_name,
        "num_items": len(records),
    }


def build_chroma_metadata(record: dict) -> dict:
    metadata = {
        "chunk_id": record["chunk_id"],
        "doc_id": record["doc_id"],
        "source_id": record["source_id"],
        "page_key": record.get("page_key") or "",
        "title": collapse_whitespace(record.get("title") or ""),
        "section_path": record.get("section_path_text") or "",
        "source_url": record.get("source_url") or "",
        "trust_level": record.get("trust_level") or "",
        "topic_tags": record.get("topic_tags_text") or "",
        "chunk_index": int(record.get("chunk_index") or 0),
        "word_count": int(record.get("word_count") or 0),
        "char_count": int(record.get("char_count") or 0),
    }
    if record.get("page_number") is not None:
        metadata["page_number"] = int(record["page_number"])
    return metadata


def build_collection_name(topic: str, snapshot_id: str) -> str:
    return f"{slugify(topic)}__{slugify(snapshot_id)}"

SOURCE_REGISTRY = [{'source_id': 'k8s_security_docs',
  'name': 'Kubernetes Security Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://kubernetes.io/docs/concepts/security/',
                'https://kubernetes.io/docs/tasks/administer-cluster/securing-a-cluster/',
                'https://kubernetes.io/docs/concepts/security/pod-security-standards/',
                'https://kubernetes.io/docs/concepts/security/pod-security-admission/',
                'https://kubernetes.io/docs/concepts/security/service-accounts/',
                'https://kubernetes.io/docs/concepts/security/controlling-access/',
                'https://kubernetes.io/docs/concepts/security/rbac-good-practices/',
                'https://kubernetes.io/docs/concepts/security/secrets-good-practices/',
                'https://kubernetes.io/docs/concepts/security/multi-tenancy/',
                'https://kubernetes.io/docs/concepts/security/linux-security/',
                'https://kubernetes.io/docs/concepts/security/linux-kernel-security-constraints/',
                'https://kubernetes.io/docs/concepts/security/hardening-guide/authentication-mechanisms/',
                'https://kubernetes.io/docs/concepts/security/api-server-bypass-risks/',
                'https://kubernetes.io/docs/concepts/security/security-checklist/',
                'https://kubernetes.io/docs/concepts/security/application-security-checklist/'],
  'allowed_domains': ['kubernetes.io'],
  'refresh_frequency': 'weekly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'hardening', 'cluster-security', 'workload-security']},
 {'source_id': 'k8s_access_control_docs',
  'name': 'Kubernetes API Access Control Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://kubernetes.io/docs/reference/access-authn-authz/',
                'https://kubernetes.io/docs/reference/access-authn-authz/authentication/',
                'https://kubernetes.io/docs/reference/access-authn-authz/authorization/',
                'https://kubernetes.io/docs/reference/access-authn-authz/rbac/',
                'https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/',
                'https://kubernetes.io/docs/reference/access-authn-authz/kubelet-authn-authz/',
                'https://kubernetes.io/docs/reference/access-authn-authz/bootstrap-tokens/',
                'https://kubernetes.io/docs/reference/access-authn-authz/kubelet-tls-bootstrapping/'],
  'allowed_domains': ['kubernetes.io'],
  'refresh_frequency': 'weekly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'authentication', 'authorization', 'admission-control', 'rbac']},
 {'source_id': 'k8s_operational_security_docs',
  'name': 'Kubernetes Operational Security Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://kubernetes.io/docs/concepts/services-networking/network-policies/',
                'https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/',
                'https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/',
                'https://kubernetes.io/docs/concepts/cluster-administration/admission-webhooks-good-practices/'],
  'allowed_domains': ['kubernetes.io'],
  'refresh_frequency': 'weekly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'network-policy', 'encryption', 'auditing', 'admission-webhooks']},
 {'source_id': 'k8s_cve_feed',
  'name': 'Kubernetes Official CVE Feed',
  'type': 'json_feed',
  'parser': 'kubernetes_cve_feed',
  'seed_urls': ['https://kubernetes.io/docs/reference/issues-security/official-cve-feed/index.json'],
  'allowed_domains': ['kubernetes.io'],
  'refresh_frequency': 'daily',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'cve', 'vulnerabilities']},
 {'source_id': 'nist_sp_800_190',
  'name': 'NIST SP 800-190 Application Container Security Guide',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-190.pdf'],
  'allowed_domains': ['nvlpubs.nist.gov'],
  'refresh_frequency': 'monthly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'containers', 'nist', 'container-security']},
 {'source_id': 'owasp_k8s_cheatsheet',
  'name': 'OWASP Kubernetes Security Cheat Sheet',
  'type': 'html',
  'parser': 'owasp_html',
  'seed_urls': ['https://cheatsheetseries.owasp.org/cheatsheets/Kubernetes_Security_Cheat_Sheet.html'],
  'allowed_domains': ['cheatsheetseries.owasp.org'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'owasp', 'checklist']},
 {'source_id': 'cncf_cloud_native_security_whitepaper',
  'name': 'CNCF Cloud Native Security Whitepaper',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://tag-security.cncf.io/community/resources/security-whitepaper/v2/CNCF_cloud-native-security-whitepaper-May2022-v2.pdf'],
  'allowed_domains': ['tag-security.cncf.io'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'cncf', 'cloud-native', 'whitepaper']},
 {'source_id': 'aalto_k8s_security_thesis',
  'name': 'Aalto Kubernetes Security Thesis',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://aaltodoc.aalto.fi/server/api/core/bitstreams/d523d51a-e37d-4df4-acc6-7f3a75f44af1/content'],
  'allowed_domains': ['aaltodoc.aalto.fi'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'academic', 'security', 'thesis', 'hardening']},
 {'source_id': 'k8s_security_observability_book',
  'name': 'Kubernetes Security and Observability Book',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://ioannisgk.com/wp-content/uploads/2023/12/Kubernetes-Security-and-Observability.pdf'],
  'allowed_domains': ['ioannisgk.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'practitioner', 'observability', 'workload-security', 'network-policy']},
 {'source_id': 'wallarm_kubernetes_security_whitepaper',
  'name': 'Wallarm 7 Steps to Kubernetes Security Whitepaper',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://cdn.prod.website-files.com/5ff66329429d880392f6cba2/60061e3fc399246789cb179c_Wallarm.%20Whitepaper.%207%20Steps%20to%20Kubernetes%20Security.pdf'],
  'allowed_domains': ['website-files.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'whitepaper', 'hardening', 'vendor']},
 {'source_id': 'practical_devsecops_kubernetes_security_101',
  'name': 'Practical DevSecOps Kubernetes Security 101',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://www.practical-devsecops.com/wp-content/uploads/2023/10/Kubernetes-Security-101.pdf'],
  'allowed_domains': ['practical-devsecops.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'devsecops', 'training', 'hardening']},
 {'source_id': 'redhat_kubernetes_security_pdf',
  'name': 'Red Hat Kubernetes Security PDF',
  'type': 'pdf',
  'parser': 'page_aware_pdf',
  'seed_urls': ['https://www.redhat.com/tracks/_pfcdn/assets/10330/contents/318568/ea1ce4f7-13d6-439d-bdf4-c804284d32ae.pdf'],
  'allowed_domains': ['redhat.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'redhat', 'security', 'hardening']},
 {'source_id': 'aws_eks_security_docs',
  'name': 'Amazon EKS Security Best Practices',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://docs.aws.amazon.com/eks/latest/best-practices/security.html',
                'https://docs.aws.amazon.com/eks/latest/best-practices/pod-security.html',
                'https://docs.aws.amazon.com/eks/latest/best-practices/network-security.html',
                'https://docs.aws.amazon.com/eks/latest/best-practices/data-encryption-and-secrets-management.html',
                'https://docs.aws.amazon.com/eks/latest/best-practices/incident-response-and-forensics.html'],
  'allowed_domains': ['docs.aws.amazon.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'eks', 'managed-kubernetes', 'security', 'hardening', 'incident-response']},
 {'source_id': 'gke_security_docs',
  'name': 'Google Kubernetes Engine Security Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://cloud.google.com/kubernetes-engine/docs/how-to/hardening-your-cluster',
                'https://docs.cloud.google.com/kubernetes-engine/docs/concepts/security-overview',
                'https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-security-in-networking'],
  'allowed_domains': ['cloud.google.com', 'docs.cloud.google.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'gke', 'managed-kubernetes', 'security', 'hardening', 'network-security']},
 {'source_id': 'aks_security_docs',
  'name': 'Azure Kubernetes Service Security Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://learn.microsoft.com/en-us/azure/aks/best-practices',
                'https://learn.microsoft.com/en-us/azure/aks/operator-best-practices-container-image-management',
                'https://learn.microsoft.com/en-us/azure/aks/network-policy-best-practices',
                'https://learn.microsoft.com/en-us/azure/aks/secure-container-access'],
  'allowed_domains': ['learn.microsoft.com'],
  'refresh_frequency': 'monthly',
  'trust_level': 'official',
  'topic_tags': ['kubernetes', 'aks', 'managed-kubernetes', 'security', 'hardening', 'image-security', 'network-policy']},
 {'source_id': 'kubescape_security_docs',
  'name': 'Kubescape Security Framework Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://kubescape.io/docs/frameworks-and-controls/',
                'https://kubescape.io/docs/controls/',
                'https://kubescape.io/docs/scanning/',
                'https://kubescape.io/docs/guides/kubescape-cli/',
                'https://kubescape.io/docs/operator/'],
  'allowed_domains': ['kubescape.io'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'kubescape', 'security-controls', 'posture-management', 'hardening']},
 {'source_id': 'gatekeeper_policy_docs',
  'name': 'OPA Gatekeeper Policy Library Docs',
  'type': 'html',
  'parser': 'kubernetes_html',
  'seed_urls': ['https://open-policy-agent.github.io/gatekeeper/website/',
                'https://open-policy-agent.github.io/gatekeeper/website/docs/library/',
                'https://open-policy-agent.github.io/gatekeeper/website/docs/security/'],
  'allowed_domains': ['open-policy-agent.github.io'],
  'refresh_frequency': 'monthly',
  'trust_level': 'trusted',
  'topic_tags': ['kubernetes', 'opa', 'gatekeeper', 'policy-as-code', 'admission-control', 'governance']}]
