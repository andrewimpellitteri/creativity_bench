"""Blinding and contextual fidelity checks for the offline reviewer packet."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_human_review.py"
SPEC = importlib.util.spec_from_file_location("human_review_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def make_run(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    transcript = [
        {
            "story": "<script>alert('story')</script>",
            "accepted": True,
            "verdict": {"evidence": "SECRET_JUDGE_EVIDENCE"},
        },
        {"story": "rejected", "accepted": False},
        {"story": "last", "accepted": True},
        {"story": None, "accepted": False},
    ]
    (runs / "model.json").write_text(
        json.dumps(
            {
                "model": "SECRET_MODEL",
                "provider": "SECRET_PROVIDER",
                "seed": 42,
                "tasks": {
                    "same_but_different": {
                        "score": 0.5,
                        "details": {"premises": [{"premise": "<fence>", "transcript": transcript}]},
                    }
                },
            }
        )
    )
    return runs


def test_context_blinding_escaping_and_blank_ratings(tmp_path):
    runs = make_run(tmp_path)
    out = tmp_path / "packet"
    assert exporter.export_packet(runs, out) == 3
    public = json.loads((out / "review_items.json").read_text())
    by_story = {item["candidate"]: item for item in public}
    assert by_story["last"]["accepted_stories"] == ["<script>alert('story')</script>"]
    assert by_story["rejected"]["accepted_stories"] == ["<script>alert('story')</script>"]
    assert by_story["<script>alert('story')</script>"]["accepted_stories"] == []
    page = (out / "review.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;fence&gt;" in page
    for filename in ("review_items.json", "review.html", "responses.csv", "INSTRUCTIONS.txt"):
        text = (out / filename).read_text()
        assert "SECRET_" not in text
        assert "model.json" not in text
    with (out / "responses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["item_id"] for row in rows} == {item["item_id"] for item in public}
    assert all(not value for row in rows for key, value in row.items() if key != "item_id")
    private = json.loads((out / "private_key.json").read_text())
    assert len(private["skipped_without_story"]) == 1
    assert len(private["items"]) == 3
    assert "SECRET_MODEL" in (out / "private_key.json").read_text()
    with pytest.raises(ValueError, match="new or empty"):
        exporter.export_packet(runs, out)


def test_controls_duplicate_occurrences_and_reproducible_order(tmp_path):
    runs = make_run(tmp_path)
    controls = tmp_path / "controls.json"
    control = {
        "id": "SECRET_CONTROL",
        "split": "development",
        "expected": {"plot_distinct": False},
        "premise": "<fence>",
        "candidate": "last",
        "accepted_stories": [],
    }
    controls.write_text(json.dumps([control, control]))
    first, second, third = [tmp_path / name for name in ("one", "two", "three")]
    exporter.export_packet(runs, first, controls, seed=7)
    exporter.export_packet(runs, second, controls, seed=7)
    exporter.export_packet(runs, third, controls, seed=8)
    assert (first / "review_items.json").read_bytes() == (second / "review_items.json").read_bytes()
    assert (first / "review_items.json").read_bytes() != (third / "review_items.json").read_bytes()
    public = json.loads((first / "review_items.json").read_text())
    assert len({item["item_id"] for item in public}) == 5
    assert "SECRET_CONTROL" not in (first / "review_items.json").read_text()
    assert all(
        set(item) == {"item_id", "premise", "candidate", "accepted_stories"} for item in public
    )


def test_empty_input_fails_without_writing(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "packet"
    with pytest.raises(ValueError, match="No reviewable"):
        exporter.export_packet(runs, out)
    assert not out.exists()
