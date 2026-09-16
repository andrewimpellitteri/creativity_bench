#!/usr/bin/env python
"""Chart Same But Different acceptance curves from a directory of saved runs.

    scripts/plot_acceptance_curves.py results/saturation-20260915/runs \
        results/saturation-20260915/ACCEPTANCE_CURVES.png

Reads only saved transcripts, so it costs nothing and can be re-run against any
cohort. A PNG output also writes a same-basename SVG copy, as elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creativity_bench.visualize import plot_acceptance_curves  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    runs_dir = argv[1]
    out = argv[2] if len(argv) == 3 else "acceptance_curves.png"
    return plot_acceptance_curves(runs_dir, out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
