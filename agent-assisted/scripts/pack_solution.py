from __future__ import annotations

"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
import json
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from a TOML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.toml"
    else:
        config_path = _resolve_project_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def resolve_solution_dir(
    language: str,
    config_path: Path | None,
    solution_dir: Path | None,
) -> Path:
    """Resolve the source directory for a root or definition-local config."""
    if solution_dir is not None:
        return _resolve_project_path(solution_dir)

    config_base = PROJECT_ROOT if config_path is None else _resolve_project_path(config_path).parent
    candidates = [
        config_base / "solution" / language,
        PROJECT_ROOT / "solution" / language,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Source directory not found. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def pack_solution(
    output_path: Path = None,
    config_path: Path | None = None,
    solution_dir: Path | None = None,
) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config(config_path)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    if language not in {"triton", "cuda"}:
        raise ValueError(f"Unsupported language: {language}")

    source_dir = resolve_solution_dir(language, config_path, solution_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    spec = dict(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        dependencies=[],
        destination_passing_style=True,
    )
    binding = build_config.get("binding")
    if binding:
        spec["binding"] = binding

    sources = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        sources.append(
            {
                "path": path.relative_to(source_dir).as_posix(),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    if not sources:
        raise ValueError(f"No source files found: {source_dir}")

    solution = {
        "name": solution_config["name"],
        "definition": solution_config["definition"],
        "author": solution_config["author"],
        "spec": spec,
        "sources": sources,
    }

    # Write to output file
    if output_path is None:
        output_base = PROJECT_ROOT if config_path is None else _resolve_project_path(config_path).parent
        output_path = output_base / "solution.json"
    else:
        output_path = _resolve_project_path(output_path)

    output_path.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution['name']}")
    print(f"  Definition: {solution['definition']}")
    print(f"  Author: {solution['author']}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=None,
        help="Explicit source directory to pack (default: infer from build.language)",
    )
    args = parser.parse_args()

    try:
        pack_solution(
            output_path=args.output,
            config_path=args.config_path,
            solution_dir=args.solution_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
