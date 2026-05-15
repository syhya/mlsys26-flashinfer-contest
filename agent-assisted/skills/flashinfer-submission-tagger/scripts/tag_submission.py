#!/usr/bin/env python3
"""Select, create, and optionally push FlashInfer-style submission tags."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


class GitError(RuntimeError):
    pass


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def list_tags(repo: Path, prefix: str) -> list[str]:
    output = run_git(repo, "tag", "--list", f"{prefix}*")
    return [line.strip() for line in output.splitlines() if line.strip()]


def compute_next_tag(tags: list[str], prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_version = 0
    for tag in tags:
        match = pattern.match(tag)
        if match:
            max_version = max(max_version, int(match.group(1)))
    return f"{prefix}{max_version + 1}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and push submission-vN tags")
    parser.add_argument("--repo", default=".", help="Git repository path")
    parser.add_argument("--prefix", default="submission-v", help="Tag prefix")
    parser.add_argument("--tag", help="Explicit tag name to use")
    parser.add_argument("--remote", default="origin", help="Remote to push to")
    parser.add_argument("--print-next", action="store_true", help="Print next available tag and exit")
    parser.add_argument("--create", action="store_true", help="Create the selected tag locally")
    parser.add_argument("--push", action="store_true", help="Push the selected tag to the remote")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    try:
        top = Path(run_git(repo, "rev-parse", "--show-toplevel"))
        commit = run_git(top, "rev-parse", "HEAD")
        branch = run_git(top, "rev-parse", "--abbrev-ref", "HEAD")
        tags = list_tags(top, args.prefix)
        selected_tag = args.tag or compute_next_tag(tags, args.prefix)

        if args.print_next and not args.create and not args.push:
            print(selected_tag)
            return 0

        if selected_tag in tags and args.create:
            raise GitError(f"tag already exists: {selected_tag}")

        remotes = {line.strip() for line in run_git(top, "remote").splitlines() if line.strip()}
        if args.push and args.remote not in remotes:
            raise GitError(f"remote not found: {args.remote}")

        if args.create:
            run_git(top, "tag", selected_tag)

        if args.push:
            run_git(top, "push", args.remote, selected_tag)

        print(f"repo={top}")
        print(f"branch={branch}")
        print(f"commit={commit}")
        print(f"tag={selected_tag}")
        print(f"created={'yes' if args.create else 'no'}")
        print(f"pushed={'yes' if args.push else 'no'}")
        return 0
    except GitError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
