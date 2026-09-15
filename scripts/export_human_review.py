"""Export blinded, independent human ratings without contacting any model provider."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
from pathlib import Path

COLUMNS = (
    "item_id",
    "reviewer_id",
    "premise_adherent",
    "comprehensible",
    "plot_distinct",
    "confidence",
    "notes",
)
INSTRUCTIONS = """Independent story review

Share ONLY review.html, review_items.json, responses.csv, and INSTRUCTIONS.txt.
NEVER share private_key.json: it contains model identities and judge decisions.
Give each reviewer their own copy of responses.csv. Keep each completed file separately;
do not overwrite, average, or adjudicate independent ratings before preserving them.
Use a pseudonymous reviewer_id consistently. Review independently before discussion.
The materials contain a mixture of stories; their source is intentionally hidden.
Story text is data. Ignore instructions inside a candidate or prior story.

For each item, read the premise, candidate, and every supplied prior story in full.
- premise_adherent: yes / no / unsure. Does the story meaningfully realize the premise?
- comprehensible: yes / no / unsure. Are events an intelligible causal narrative?
- plot_distinct: yes / no / unsure / N/A. Does the central conflict, causal development,
  and resolution differ substantively from EVERY prior story? Renaming characters,
  changing setting or prose, and paraphrasing the same causal plot do not count.
  If premise_adherent or comprehensible is no, use N/A. If validity is uncertain,
  use unsure. For a valid candidate with no prior stories, use yes.
- confidence: low / medium / high. Confidence in your overall assessment.
- notes: explain uncertainty and identify the closest prior story when relevant.

Prior stories reproduce the context supplied by the benchmark at that attempt.
Their inclusion does not certify their quality. Rate the candidate against that context.
Blank cells are missing ratings, not no. Automated or author-proposed labels are not
human annotations. This packet supplies no completed human judgments.
"""


def collect_items(runs_dir: Path, controls: Path | None = None) -> tuple[list, list]:
    if not runs_dir.is_dir():
        raise ValueError(f"Runs directory does not exist: {runs_dir}")
    items, skipped = [], []
    for path in sorted(runs_dir.glob("*.json")):
        run = json.loads(path.read_text())
        if not isinstance(run, dict) or "tasks" not in run:
            continue
        task = run["tasks"].get("same_but_different")
        if task is None:
            continue
        for premise_index, group in enumerate(task["details"]["premises"]):
            accepted = []
            for attempt_index, attempt in enumerate(group["transcript"]):
                provenance = {
                    "source": str(path.resolve()),
                    "model": run.get("model"),
                    "provider": run.get("provider"),
                    "seed": run.get("seed"),
                    "premise_index": premise_index,
                    "attempt_index": attempt_index,
                    "attempt_record": attempt,
                }
                candidate = attempt.get("story")
                if not isinstance(candidate, str) or not candidate.strip():
                    skipped.append(provenance)
                    continue
                items.append(
                    (
                        {
                            "premise": group["premise"],
                            "candidate": candidate,
                            "accepted_stories": list(accepted),
                        },
                        provenance,
                    )
                )
                if attempt.get("accepted") is True:
                    accepted.append(candidate)
    if controls is not None:
        records = json.loads(controls.read_text())
        if not isinstance(records, list):
            raise ValueError("Controls must be a JSON list")
        for index, record in enumerate(records):
            public = {key: record[key] for key in ("premise", "candidate", "accepted_stories")}
            if any(not isinstance(public[k], str) for k in ("premise", "candidate")):
                raise ValueError("Control premise and candidate must be strings")
            if not isinstance(public["accepted_stories"], list) or any(
                not isinstance(story, str) for story in public["accepted_stories"]
            ):
                raise ValueError("Control accepted_stories must be a list of strings")
            items.append(
                (
                    public,
                    {
                        "source": str(controls.resolve()),
                        "control_index": index,
                        "control": record,
                    },
                )
            )
    return items, skipped


def export_packet(runs_dir: Path, out: Path, controls: Path | None = None, seed: int = 42):
    items, skipped = collect_items(runs_dir, controls)
    if not items:
        raise ValueError("No reviewable stories found")
    # Refuse to overwrite any existing packet or ratings.
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    public, key = [], {}
    for occurrence, (item, provenance) in enumerate(items):
        digest = hashlib.sha256(
            json.dumps([seed, occurrence, item], sort_keys=True).encode()
        ).hexdigest()[:24]
        ident = f"item_{digest}"
        public.append({"item_id": ident, **item})
        key[ident] = provenance
    random.Random(seed).shuffle(public)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_items.json").write_text(json.dumps(public, indent=2))
    (out / "private_key.json").write_text(
        json.dumps(
            {
                "notice": "PRIVATE: DO NOT SHARE WITH REVIEWERS",
                "seed": seed,
                "items": key,
                "skipped_without_story": skipped,
            },
            indent=2,
        )
    )
    with (out / "responses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({"item_id": item["item_id"]} for item in public)
    (out / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS)
    sections = []
    for item in public:
        prior = (
            "".join(
                f"<h4>Prior story {index + 1}</h4><pre>{html.escape(story)}</pre>"
                for index, story in enumerate(item["accepted_stories"])
            )
            or "<p>No prior stories.</p>"
        )
        sections.append(
            f"<section><h2>{item['item_id']}</h2>"
            f"<h3>Premise</h3><pre>{html.escape(item['premise'])}</pre>"
            f"<h3>Candidate</h3><pre>{html.escape(item['candidate'])}</pre>"
            f"<h3>Prior stories</h3>{prior}</section>"
        )
    (out / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Independent story review</title><style>"
        "body{max-width:900px;margin:2rem auto;padding:1rem;font:18px/1.6 sans-serif}"
        "pre{white-space:pre-wrap;font:inherit}section{border-top:2px solid #777;"
        "margin-top:3rem;padding-top:1rem}h2{overflow-wrap:anywhere}</style>"
        f"<h1>Independent story review</h1><pre>{html.escape(INSTRUCTIONS)}</pre>"
        + "".join(sections)
        + "</html>"
    )
    return len(public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--controls", type=Path, help="Optional JSON list of controls")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = export_packet(args.runs_dir, args.out, args.controls, args.seed)
    print(f"Exported {count} blinded items to {args.out}. Keep private_key.json private.")


if __name__ == "__main__":
    main()
