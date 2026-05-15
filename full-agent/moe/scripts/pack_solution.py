"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Python, Triton, or CUDA) into a Solution JSON file for submission.
"""

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files
except ImportError:
    from pydantic import BaseModel, Field, model_validator

    class SupportedLanguages(str, Enum):
        PYTHON = "python"
        TRITON = "triton"
        CPP = "cpp"
        CUDA = "cuda"
        TILELANG = "tilelang"

    class SupportedBindings(str, Enum):
        TVM_FFI = "tvm-ffi"
        TORCH = "torch"

    class SourceFile(BaseModel):
        path: str
        content: str

        @model_validator(mode="after")
        def validate_path(self):
            src_path = Path(self.path)
            if src_path.is_absolute():
                raise ValueError(f"Invalid source path (absolute path not allowed): {self.path}")
            if ".." in src_path.parts:
                raise ValueError(
                    f"Invalid source path (parent directory traversal not allowed): {self.path}"
                )
            return self

    class BuildSpec(BaseModel):
        language: SupportedLanguages
        target_hardware: List[str] = Field(min_length=1)
        entry_point: str
        dependencies: List[str] = Field(default_factory=list)
        destination_passing_style: bool = True
        binding: Optional[SupportedBindings] = None

        @model_validator(mode="after")
        def validate_entry_point(self):
            if self.entry_point.count("::") != 1:
                raise ValueError(
                    f'Invalid entry point format: {self.entry_point}. Expected "<file_path>::<function_name>".'
                )
            return self

    class Solution(BaseModel):
        name: str
        definition: str
        author: str
        spec: BuildSpec
        sources: List[SourceFile] = Field(min_length=1)
        description: str = ""

        @model_validator(mode="after")
        def validate_entry_source(self):
            entry_file = self.spec.entry_point.split("::")[0]
            seen_paths = {source.path for source in self.sources}
            if entry_file not in seen_paths:
                raise ValueError(f"Entry source file '{entry_file}' not found in sources")
            return self

    VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}

    def pack_solution_from_files(
        path: str, spec: BuildSpec, name: str, definition: str, author: str, description: str = ""
    ) -> Solution:
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        sources = []
        for file_path in sorted(path_obj.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
                continue
            sources.append(
                SourceFile(path=file_path.name, content=file_path.read_text(encoding="utf-8"))
            )

        if not sources:
            raise ValueError(f"No source files found in directory: {path}")

        return Solution(
            name=name,
            definition=definition,
            author=author,
            description=description,
            spec=spec,
            sources=sources,
        )


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]
    binding = build_config.get("binding")
    dependencies = build_config.get("dependencies", [])

    # Determine source directory based on language
    if language == "python":
        source_dir = PROJECT_ROOT / "solution" / "python"
    elif language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps_default = False if language == "python" else True
    dps = build_config.get("destination_passing_style", dps_default)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        dependencies=dependencies,
        destination_passing_style=dps,
        binding=binding,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2), encoding="utf-8")
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")
    print(f"  Entry point: {entry_point}")
    if binding:
        print(f"  Binding: {binding}")

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
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
