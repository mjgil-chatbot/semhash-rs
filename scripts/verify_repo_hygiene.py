#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
FORBIDDEN_SUBSTRINGS = ("".join(("/", "home", "/")), "".join(("~", "/")))
SCAN_PATHS = (
    ROOT / "README.md",
    ROOT / "Cargo.toml",
    ROOT / "LICENSE",
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / ".github",
)
REQUIRED_FILES = (
    ROOT / "LICENSE",
    ROOT / "THIRD_PARTY_LICENSE",
)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_PATHS:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    return files


def main() -> int:
    failures: list[str] = []

    for required in REQUIRED_FILES:
        if not required.exists():
            failures.append(f"missing required file: {required.relative_to(ROOT)}")

    for path in iter_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in text:
                failures.append(f"{rel}: contains forbidden path fragment {forbidden!r}")
        if EMAIL_RE.search(text):
            failures.append(f"{rel}: contains an email address")

    if failures:
        print("repo hygiene check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("repo hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
