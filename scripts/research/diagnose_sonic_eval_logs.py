#!/usr/bin/env python3
"""Summarize SONIC eval-log health for metric-completeness debugging.

This is deliberately diagnostic-only: it never fabricates metrics and only reports
strings that are actually present in the eval log.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_ALL_MPJPE_RE = re.compile(r"^All:\s+.*\bmpjpe_(?:g|l|pa)\b", re.IGNORECASE)
_ANY_MPJPE_RE = re.compile(r"\bmpjpe(?:_[a-z0-9_]+)?\b", re.IGNORECASE)
_TRACEBACK_RE = re.compile(r"Traceback|Error executing job|RuntimeError|Exception|AssertionError")
_TIMEOUT_RE = re.compile(r"Command timed out|TimeoutExpired|timed out after", re.IGNORECASE)
_SEQUENCE_STARTED_RE = re.compile(r"Sequence progress:")
_SEQUENCE_COMPLETED_RE = re.compile(r"Sequence progress:\s*100%|!{7}|\bAll:\s+")
_RETURN_CODE_RE = re.compile(r"(?:^|\b)(?:rc|returncode|exit_code)\s*[=:]\s*(-?\d+)", re.IGNORECASE)
_METRIC_LINE_RE = re.compile(r"^(?:All:|Succ:|Success Rate:|Progress Rate:).*", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _last_lines_path(eval_log: Path, output_dir: Path, variant: str, lines: list[str], count: int = 30) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{variant}_last_{count}_eval_lines.txt"
    path.write_text("\n".join(lines[-count:]) + ("\n" if lines else ""), encoding="utf-8")
    return path


def diagnose_eval_log(variant: str, eval_log: Path, output_dir: Path) -> dict[str, Any]:
    text = _read_text(eval_log)
    lines = text.splitlines()
    candidate_metric_lines = [line.strip() for line in lines if _METRIC_LINE_RE.search(line.strip())]
    return_code_matches = _RETURN_CODE_RE.findall(text)
    last_30_path = _last_lines_path(eval_log, output_dir, variant, lines)

    return {
        "variant": variant,
        "eval_log_path": str(eval_log),
        "eval_return_code": int(return_code_matches[-1]) if return_code_matches else None,
        "eval_traceback": bool(_TRACEBACK_RE.search(text)),
        "eval_timeout": bool(_TIMEOUT_RE.search(text)),
        "contains_all_mpjpe": any(_ALL_MPJPE_RE.search(line.strip()) for line in lines),
        "contains_any_mpjpe": bool(_ANY_MPJPE_RE.search(text)),
        "last_30_log_lines_path": str(last_30_path),
        "num_sequences_started": len(_SEQUENCE_STARTED_RE.findall(text)),
        "num_sequences_completed": len(_SEQUENCE_COMPLETED_RE.findall(text)),
        "candidate_metric_lines": candidate_metric_lines[-20:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-log", action="append", nargs=2, metavar=("VARIANT", "LOG"), required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_json = Path(args.output_json)
    diag_dir = output_json.parent / "eval_diagnostics_last_lines"
    diagnostics = [
        diagnose_eval_log(variant, Path(log_path), diag_dir)
        for variant, log_path in args.variant_log
    ]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote eval diagnostics to {output_json}")


if __name__ == "__main__":
    main()
