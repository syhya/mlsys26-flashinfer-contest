#!/usr/bin/env python3
"""Validate FlashInfer contest config.toml topology before submission work."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


IGNORED_FILE_NAMES = {".DS_Store"}


class TopologyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedConfig:
    config_path: Path
    base_dir: Path
    language: str
    source_dir: Path
    entry_file: Path
    definition: str
    name: str


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def is_real_source_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if "__pycache__" in path.parts:
        return False
    if path.name in IGNORED_FILE_NAMES:
        return False
    return True


def resolve_config(config_path: Path, repo: Path) -> ResolvedConfig:
    cfg = load_toml(config_path)
    solution = cfg["solution"]
    build = cfg["build"]
    language = build["language"]
    entry_point = build["entry_point"]
    entry_file = entry_point.split("::", 1)[0]
    base_dir = config_path.parent
    if base_dir == repo:
        source_dir = repo / "solution" / language
    else:
        source_dir = base_dir / "solution" / language
    return ResolvedConfig(
        config_path=config_path,
        base_dir=base_dir,
        language=language,
        source_dir=source_dir,
        entry_file=source_dir / entry_file,
        definition=solution["definition"],
        name=solution["name"],
    )


def discover_configs(repo: Path) -> tuple[Path | None, list[Path], list[Path]]:
    root_cfg = repo / "config.toml"
    root = root_cfg if root_cfg.exists() else None

    child_cfgs: list[Path] = []
    invalid_cfgs: list[Path] = []

    for path in sorted(repo.glob("*/config.toml")):
        child_cfgs.append(path)

    root_solution = repo / "solution"
    if root_solution.exists():
        invalid_cfgs.extend(sorted(root_solution.rglob("config.toml")))

    for child_cfg in child_cfgs:
        solution_root = child_cfg.parent / "solution"
        if solution_root.exists():
            invalid_cfgs.extend(sorted(solution_root.rglob("config.toml")))

    return root, child_cfgs, invalid_cfgs


def ensure_real_sources(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise TopologyError(f"resolved source dir does not exist: {source_dir}")
    files = [p for p in sorted(source_dir.rglob("*")) if is_real_source_file(p)]
    if not files:
        raise TopologyError(f"resolved source dir has zero real source files: {source_dir}")
    return files


def validate(repo: Path) -> tuple[str, list[ResolvedConfig]]:
    root_cfg, child_cfgs, invalid_cfgs = discover_configs(repo)

    if invalid_cfgs:
        rels = ", ".join(str(p.relative_to(repo)) for p in invalid_cfgs)
        raise TopologyError(f"invalid config.toml location(s): {rels}")

    if root_cfg and child_cfgs:
        raise TopologyError(
            "root config.toml and definition-subdir configs coexist; choose exactly one layout"
        )

    if root_cfg:
        resolved = [resolve_config(root_cfg, repo)]
        layout = "single-definition-root"
    elif child_cfgs:
        resolved = [resolve_config(path, repo) for path in child_cfgs]
        layout = "definition-subdir-single" if len(child_cfgs) == 1 else "definition-subdir-multi"
    else:
        raise TopologyError("no config.toml found at the repo root or definition subdirectories")

    if layout.startswith("definition-subdir") and root_cfg:
        raise TopologyError("multi-definition repos must not keep a root config.toml")

    seen_definitions: dict[str, Path] = {}
    for cfg in resolved:
        if cfg.definition in seen_definitions:
            prev = seen_definitions[cfg.definition]
            raise TopologyError(
                f"duplicate definition '{cfg.definition}' in {prev.relative_to(repo)} and "
                f"{cfg.config_path.relative_to(repo)}"
            )
        seen_definitions[cfg.definition] = cfg.config_path

        ensure_real_sources(cfg.source_dir)
        if not cfg.entry_file.exists():
            raise TopologyError(
                "entry_point file is missing relative to the resolved source dir: "
                f"{cfg.entry_file.relative_to(repo)}"
            )

    return layout, resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    try:
        layout, resolved = validate(repo)
    except TopologyError as exc:
        print(f"topology=invalid", file=sys.stderr)
        print(f"repo={repo}", file=sys.stderr)
        print(f"error={exc}", file=sys.stderr)
        return 1

    print(f"topology={layout}")
    print(f"repo={repo}")
    for cfg in resolved:
        print(f"config={cfg.config_path.relative_to(repo)}")
        print(f"definition={cfg.definition}")
        print(f"name={cfg.name}")
        print(f"source_dir={cfg.source_dir.relative_to(repo)}")
        print(f"entry_file={cfg.entry_file.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
