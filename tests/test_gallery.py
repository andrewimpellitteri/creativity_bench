import json

import pytest

from creativity_bench.gallery import write_gallery


def test_gallery_escapes_saved_content_and_shows_review_fields(tmp_path):
    attack = '<script>alert("story")</script>'
    run = {
        "model": attack,
        "tasks": {
            "same_but_different": {
                "details": {
                    "judge_model": "judge",
                    "protocol": "v1",
                    "premises": [
                        {
                            "premise": attack,
                            "acceptance_curve": [1, 1],
                            "transcript": [
                                {
                                    "attempt": 1,
                                    "accepted": False,
                                    "story": attack,
                                    "status": "ok",
                                    "rejection_reasons": ["plot_distinct"],
                                    "verdict": {"evidence": attack, "summary": "Same causal plot"},
                                    "judge_responses": [attack],
                                }
                            ],
                        }
                    ],
                }
            }
        },
    }
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run))
    output = write_gallery(source, tmp_path / "nested" / "gallery.html")
    page = output.read_text()
    assert attack not in page
    assert "&lt;script&gt;" in page
    assert "<script" not in page
    assert "Exploratory judgments" in page
    assert "Same causal plot" in page
    assert "plot_distinct" in page
    assert "Judgment status" in page
    assert "<svg" in page
    assert "Cumulative accepted stories: 1, 1" in page
    assert "Attempt 1 · Rejected" in page


def test_gallery_requires_task(tmp_path):
    source = tmp_path / "run.json"
    source.write_text('{"tasks": {}}')
    with pytest.raises(ValueError, match="no same_but_different"):
        write_gallery(source, tmp_path / "out.html")
