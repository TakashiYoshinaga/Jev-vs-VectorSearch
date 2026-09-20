import json
from pathlib import Path

import pytest

from app.main import PRESETS, chunks, metrics


def test_chunks_overlap_and_validation():
    assert chunks("abcdefghij", size=6, overlap=2) == ["abcdef", "efghij", "ij"]
    with pytest.raises(ValueError):
        chunks("text", size=4, overlap=4)


def test_threshold_is_inclusive_and_metrics_are_correct():
    result = metrics({"a": 0.5, "b": 0.8, "c": 0.2}, 0.5, {"a", "c"})
    assert result["tp"] == ["a"]
    assert result["fp"] == ["b"]
    assert result["fn"] == ["c"]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_zero_denominators_are_safe():
    result = metrics({}, 0.5, set())
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_all_ground_truth_ids_exist():
    dataset = Path(__file__).resolve().parents[1] / "dataset" / "documents.jsonl"
    ids = {json.loads(line)["doc_id"] for line in dataset.read_text(encoding="utf-8").splitlines()}
    for preset in PRESETS:
        assert len(preset["ground_truth"]) == len(set(preset["ground_truth"]))
        assert set(preset["ground_truth"]) <= ids
