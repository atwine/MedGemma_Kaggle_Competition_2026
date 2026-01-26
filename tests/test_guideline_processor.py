from __future__ import annotations

from pathlib import Path

import pytest

from modules.guideline_processor import process_guidelines


def test_process_guidelines_smoke() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    consolidated = data_dir / "Consolidated-HIV-and-AIDS-Guidelines-20230516.pdf"
    legacy = data_dir / "Uganda Clinical Guidelines 2023.pdf"

    # Rationale: prefer the newer consolidated guideline PDF when present, but keep a
    # fallback to the legacy filename used by the initial demo.
    pdf_path = consolidated if consolidated.exists() else legacy

    if not pdf_path.exists():
        pytest.skip("Local guideline PDF not present")

    # Rationale: limit pages to keep the smoke test fast on typical laptops.
    chunks = process_guidelines(pdf_path, max_pages=2)

    assert len(chunks) > 0
    assert all(c.page_number in (1, 2) for c in chunks)
    assert all(c.text.strip() for c in chunks)
