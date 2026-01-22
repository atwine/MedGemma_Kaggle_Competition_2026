from __future__ import annotations

from pathlib import Path

import pytest

from modules.guideline_processor import process_guidelines


def test_process_guidelines_smoke() -> None:
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "Data"
        / "Uganda Clinical Guidelines 2023.pdf"
    )

    if not pdf_path.exists():
        pytest.skip("Local guideline PDF not present")

    # Rationale: limit pages to keep the smoke test fast on typical laptops.
    chunks = process_guidelines(pdf_path, max_pages=2)

    assert len(chunks) > 0
    assert all(c.page_number in (1, 2) for c in chunks)
    assert all(c.text.strip() for c in chunks)
