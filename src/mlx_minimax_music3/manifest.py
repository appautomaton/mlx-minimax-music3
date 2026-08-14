"""Versioned checkpoint manifest schema and integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

MANIFEST_FORMAT = "mlx-minimax-music3"
MANIFEST_VERSION = 1
MAPPING_VERSION = 1


class ManifestError(ValueError):
    """Raised when a converted checkpoint manifest is invalid."""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without mapping its full contents into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("Manifest paths must be non-empty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"Manifest path must stay inside its root: {value!r}")
    return str(path)


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{key!r} must be an integer")
    return value


def _optional_integer(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{key!r} must be an integer or null")
    return value


@dataclass(frozen=True, slots=True)
class ManifestFile:
    path: str
    size: int
    sha256: str
    tensor_count: int = 0
    dtypes: tuple[str, ...] = ()
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if self.source_path is not None:
            object.__setattr__(
                self, "source_path", _relative_path(self.source_path)
            )
        if self.size < 0 or self.tensor_count < 0:
            raise ManifestError("File sizes and tensor counts cannot be negative")
        for name, digest in (
            ("sha256", self.sha256),
            ("source_sha256", self.source_sha256),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ManifestError(f"{name} must be a lowercase SHA-256 digest")
        if any(not isinstance(dtype, str) or not dtype for dtype in self.dtypes):
            raise ManifestError("dtypes must contain non-empty strings")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_dtypes = data.get("dtypes", [])
        if not isinstance(raw_dtypes, list):
            raise ManifestError("dtypes must be an array")
        return cls(
            path=_relative_path(data.get("path")),
            size=_integer(data, "size"),
            sha256=str(data.get("sha256", "")),
            tensor_count=_integer(data, "tensor_count"),
            dtypes=tuple(raw_dtypes),
            source_path=(
                None
                if data.get("source_path") is None
                else _relative_path(data["source_path"])
            ),
            source_sha256=(
                None
                if data.get("source_sha256") is None
                else str(data["source_sha256"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    name: str
    files: tuple[ManifestFile, ...]

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ManifestError(f"Invalid component name: {self.name!r}")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ManifestError(f"Duplicate files in component {self.name!r}")

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> Self:
        raw_files = data.get("files")
        if not isinstance(raw_files, list):
            raise ManifestError(f"Component {name!r} files must be an array")
        return cls(
            name=name,
            files=tuple(ManifestFile.from_dict(file) for file in raw_files),
        )


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    profile: str
    source_repository: str
    source_revision: str
    components: tuple[ComponentManifest, ...]
    quantized_modules: tuple[str, ...] = ()
    quantization_mode: str | None = None
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    format: str = MANIFEST_FORMAT
    format_version: int = MANIFEST_VERSION
    mapping_version: int = MAPPING_VERSION

    def __post_init__(self) -> None:
        if self.format != MANIFEST_FORMAT:
            raise ManifestError(f"Unsupported manifest format: {self.format!r}")
        if self.format_version != MANIFEST_VERSION:
            raise ManifestError(
                f"Unsupported manifest version: {self.format_version}"
            )
        if self.mapping_version != MAPPING_VERSION:
            raise ManifestError(
                f"Unsupported tensor mapping version: {self.mapping_version}"
            )
        if self.profile not in {"dense", "q8"}:
            raise ManifestError(f"Unsupported checkpoint profile: {self.profile!r}")
        if not self.source_repository or not self.source_revision:
            raise ManifestError("Source repository and revision are required")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ManifestError("Component names must be unique")
        files = [file.path for component in self.components for file in component.files]
        if len(files) != len(set(files)):
            raise ManifestError("Manifest file paths must be globally unique")
        if self.profile == "dense":
            if (
                self.quantized_modules
                or self.quantization_mode is not None
                or self.quantization_bits is not None
                or self.quantization_group_size is not None
            ):
                raise ManifestError("Dense manifests cannot declare quantization")
        else:
            if (
                self.quantization_mode != "affine"
                or self.quantization_bits != 8
                or self.quantization_group_size is None
                or isinstance(self.quantization_group_size, bool)
                or not isinstance(self.quantization_group_size, int)
                or self.quantization_group_size <= 0
                or not self.quantized_modules
            ):
                raise ManifestError("q8 manifests require a complete affine policy")
            if (
                any(
                    not isinstance(module, str) or not module
                    for module in self.quantized_modules
                )
                or len(self.quantized_modules) != len(set(self.quantized_modules))
            ):
                raise ManifestError(
                    "Quantized modules must be unique non-empty strings"
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["components"] = {
            component.name: {
                "files": [asdict(file) for file in component.files]
            }
            for component in self.components
        }
        result["source"] = {
            "repository": result.pop("source_repository"),
            "revision": result.pop("source_revision"),
        }
        result["quantization"] = {
            "mode": result.pop("quantization_mode"),
            "bits": result.pop("quantization_bits"),
            "group_size": result.pop("quantization_group_size"),
            "modules": result.pop("quantized_modules"),
        }
        if self.profile == "dense":
            result.pop("quantization")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_source = data.get("source")
        raw_components = data.get("components")
        if not isinstance(raw_source, dict):
            raise ManifestError("source must be an object")
        if not isinstance(raw_components, dict):
            raise ManifestError("components must be an object")
        quantization = data.get("quantization", {})
        if not isinstance(quantization, dict):
            raise ManifestError("quantization must be an object")
        modules = quantization.get("modules", [])
        if not isinstance(modules, list):
            raise ManifestError("quantization modules must be an array")
        return cls(
            format=str(data.get("format", "")),
            format_version=_integer(data, "format_version"),
            mapping_version=_integer(data, "mapping_version"),
            profile=str(data.get("profile", "")),
            source_repository=str(raw_source.get("repository", "")),
            source_revision=str(raw_source.get("revision", "")),
            components=tuple(
                ComponentManifest.from_dict(name, component)
                for name, component in sorted(raw_components.items())
            ),
            quantized_modules=tuple(modules),
            quantization_mode=quantization.get("mode"),
            quantization_bits=_optional_integer(quantization, "bits"),
            quantization_group_size=_optional_integer(
                quantization, "group_size"
            ),
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ManifestError(f"Missing checkpoint manifest: {path}") from error
        except json.JSONDecodeError as error:
            raise ManifestError(f"Invalid checkpoint manifest JSON: {path}") from error
        if not isinstance(data, dict):
            raise ManifestError("Manifest root must be an object")
        return cls.from_dict(data)

    def write(self, path: str | Path) -> None:
        """Atomically write canonical manifest JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def verify(self, root: str | Path, *, digests: bool = True) -> None:
        """Verify every declared file before model construction."""

        root = Path(root).resolve()
        for component in self.components:
            for record in component.files:
                path = (root / record.path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ManifestError(f"Missing manifest file: {record.path}")
                if path.stat().st_size != record.size:
                    raise ManifestError(f"File size mismatch: {record.path}")
                if digests and sha256_file(path) != record.sha256:
                    raise ManifestError(f"SHA-256 mismatch: {record.path}")
