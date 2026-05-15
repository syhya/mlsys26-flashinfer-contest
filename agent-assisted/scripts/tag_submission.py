#!/usr/bin/env python3
"""Create and optionally push contest submission tags."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"^submission-v(\d+)$")


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def list_submission_tags(repo: Path) -> list[str]:
    output = run_git(repo, "tag", "--list", "submission-v*")
    tags = [line.strip() for line in output.splitlines() if line.strip()]
    return sorted(tags, key=tag_version)


def tag_version(tag: str) -> int:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"Unsupported submission tag format: {tag}")
    return int(match.group(1))


def next_submission_tag(repo: Path) -> str:
    tags = [tag for tag in list_submission_tags(repo) if TAG_RE.fullmatch(tag)]
    if not tags:
        return "submission-v1"
    return f"submission-v{tag_version(tags[-1]) + 1}"


def ensure_repo(repo: Path) -> None:
    try:
        run_git(repo, "rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Not a git repository: {repo}") from exc


def create_tag(repo: Path, tag: str) -> None:
    existing = run_git(repo, "tag", "--list", tag)
    if existing:
        raise SystemExit(f"Tag already exists: {tag}")
    subprocess.run(["git", "-C", str(repo), "tag", tag], check=True)


def push_tag(repo: Path, tag: str, remote: str) -> None:
    subprocess.run(["git", "-C", str(repo), "push", remote, tag], check=True)


def current_head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository path")
    parser.add_argument("--tag", help="Explicit tag name to use")
    parser.add_argument("--remote", default="origin", help="Remote name for --push")
    parser.add_argument("--print-next", action="store_true", help="Print the next submission tag")
    parser.add_argument("--create", action="store_true", help="Create the chosen tag on HEAD")
    parser.add_argument("--push", action="store_true", help="Push the chosen tag to the remote")
    args = parser.parse_args()

    repo = args.repo.resolve()
    ensure_repo(repo)

    if args.tag is not None and not TAG_RE.fullmatch(args.tag):
        raise SystemExit("Explicit tag must match submission-vN")

    tag = args.tag or next_submission_tag(repo)

    if args.print_next:
        print(tag)

    if args.create:
        create_tag(repo, tag)

    if args.push:
        if not args.create:
            raise SystemExit("--push requires --create")
        push_tag(repo, tag, args.remote)

    if args.create or args.push:
        print(f"tag={tag}")
        print(f"commit={current_head(repo)}")

    if not any((args.print_next, args.create, args.push)):
        parser.print_help(sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
