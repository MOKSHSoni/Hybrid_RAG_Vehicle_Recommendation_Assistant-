import ast
import hashlib
from pathlib import Path

import config
from src.ingestion.csv_loader import load_csv, load_documents

EV_NAMES = [
    "Mercedes Benz EQC",
    "Jaguar I-Pace",
    "MG ZS EV",
    "Hyundai Kona Electric",
    "Tata Tigor EV",
    "Tata Nexon EV",
]
SEATING_SENTINEL_NAMES = ["Mercedes Benz CLS", "Tata Nexon EV"]


def test_loads_expected_shape():
    df = load_csv(config.RAW_CSV_PATH)
    assert df.shape == (150, 34)


def test_no_bom_in_first_column():
    df = load_csv(config.RAW_CSV_PATH)
    assert df.columns[0] == "Name"


def test_original_csv_never_mutated():
    before = _hash_file(config.RAW_CSV_PATH)
    load_documents(config.RAW_CSV_PATH, source_name="cars_cleaned")
    after = _hash_file(config.RAW_CSV_PATH)
    assert before == after


def test_150_unique_doc_ids(normalized_documents):
    doc_ids = [d.doc_id for d in normalized_documents]
    assert len(doc_ids) == 150
    assert len(set(doc_ids)) == 150


def test_seating_sentinel_rows_become_none(normalized_documents):
    for name in SEATING_SENTINEL_NAMES:
        doc = _find_by_name(normalized_documents, name)
        assert doc.raw_fields["Seating Capacity"] is None


def test_ev_engine_fields_become_none(normalized_documents):
    for name in EV_NAMES:
        doc = _find_by_name(normalized_documents, name)
        assert doc.raw_fields["ENGINE"] is None
        assert doc.raw_fields["Engine_Min_cc"] is None
        assert doc.raw_fields["Engine_Max_cc"] is None


def test_generic_type_coercion(normalized_documents):
    macan = _find_by_name(normalized_documents, "Porsche Macan")
    assert macan.raw_fields["Has_Automatic"] is True
    assert macan.raw_fields["Has_Manual"] is False
    assert isinstance(macan.raw_fields["Price_Lakhs"], float)
    assert isinstance(macan.raw_fields["Seating Capacity"], int)
    assert macan.raw_fields["Seating Capacity"] == 5


def test_messy_compound_text_stays_as_string(normalized_documents):
    macan = _find_by_name(normalized_documents, "Porsche Macan")
    assert isinstance(macan.raw_fields["EMI"], str)  # "1,18,950" -- comma breaks float()
    assert isinstance(macan.raw_fields["Peak Power"], str)  # "251 BHP@5000 RPM"


def test_ingestion_module_never_imports_enrichment():
    import src.ingestion.csv_loader as csv_loader_module
    import src.ingestion.models as models_module
    import src.ingestion.sentinels as sentinels_module

    for module in (csv_loader_module, models_module, sentinels_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "enrichment" not in node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "enrichment" not in alias.name


def _find_by_name(documents, name):
    for d in documents:
        if d.raw_fields.get("Name") == name:
            return d
    raise AssertionError(f"vehicle not found: {name}")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
