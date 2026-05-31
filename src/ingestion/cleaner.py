"""Cleaner module for HTML stripping and text normalization using multiprocessing with safe Windows fallbacks."""
import re
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any


def clean_html(text: str) -> str:
    """Remove HTML tags and normalize spacing while preserving structural newlines

    necessary for law chunking (e.g., '\nĐiều', '\nKhoản').
    """
    if not text:
        return ""
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Replace common HTML line break tags with actual newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*.*?>", "\n", text, flags=re.IGNORECASE)
    # Remove other HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML characters
    text = (
        text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Standardize spaces but preserve structural lines
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)
    
    # Return reconstructed text with single-spaced lines and preserved newlines
    return "\n".join(lines)


def clean_doc_task(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean the raw content of a single document dictionary."""
    doc_copy = doc.copy()
    doc_copy["raw_content"] = clean_html(doc_copy.get("raw_content", ""))
    return doc_copy


def clean_documents_parallel(documents: List[Dict[str, Any]], max_workers: int = None) -> List[Dict[str, Any]]:
    """Clean documents' content sequentially.

    Using sequential cleaning prevents Windows multiprocessing spawn bugs and heavy memory overhead.
    """
    return [clean_doc_task(doc) for doc in documents]
