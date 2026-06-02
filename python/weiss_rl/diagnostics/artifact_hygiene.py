from __future__ import annotations

import csv
import re
import subprocess
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

TEXT_FILE_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"})
IMAGE_FILE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tif", ".tiff"})
FORBIDDEN_BINARY_SUFFIXES = frozenset({".psd", ".ai", ".eps"})
REPO_SCAN_SKIP_DIRS = frozenset({".git", ".github", "docs"})
SUSPICIOUS_IMAGE_PATH_MARKERS = (
    "logo",
    "logos",
    "brandmark",
    "wordmark",
    "icon",
    "icons",
    "card_art",
    "card-art",
    "cardart",
    "card_face",
    "card-face",
    "cardface",
    "artwork",
    "splash",
    "thumbnail",
    "avatar",
)
SUSPICIOUS_TEXT_PATH_MARKERS = (
    "card_text",
    "card_texts",
    "card-text",
    "effect_text",
    "effect_texts",
    "effect-text",
    "ability_text",
    "ability_texts",
    "ability-text",
    "flavor_text",
    "flavor_texts",
    "flavour_text",
    "flavour_texts",
    "card_names",
    "card-names",
)
CARD_TEXT_FIELD_NAMES = frozenset(
    {
        "card_text",
        "effect_text",
        "ability_text",
        "flavor_text",
        "flavour_text",
        "card_name",
        "character_name",
        "series_name",
        "franchise_name",
    }
)
TRADEMARK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("weiss schwarz", re.compile(r"\bweiss\s*schwarz\b", re.IGNORECASE)),
    ("hololive", re.compile(r"\bhololive(?:\s+production)?\b", re.IGNORECASE)),
    ("bang dream", re.compile(r"\bbang\s+dream!?\b", re.IGNORECASE)),
    ("love live", re.compile(r"\blove\s+live!?\b", re.IGNORECASE)),
    ("project sekai", re.compile(r"\bproject\s+sekai\b", re.IGNORECASE)),
    ("the idolm@ster", re.compile(r"\bthe\s+idolm@ster\b", re.IGNORECASE)),
    ("star wars", re.compile(r"\bstar\s+wars\b", re.IGNORECASE)),
    ("marvel", re.compile(r"\bmarvel\b", re.IGNORECASE)),
    ("disney", re.compile(r"\bdisney\b", re.IGNORECASE)),
)


@dataclass(frozen=True, slots=True)
class Finding:
    surface: str
    path: str
    rule: str
    detail: str

    def format(self) -> str:
        return f"[{self.surface}] {self.path}: {self.rule}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ScanSummary:
    findings: tuple[Finding, ...]
    repo_file_count: int
    artifact_file_count: int
    replay_bundle_count: int


@dataclass(frozen=True, slots=True)
class _ScanStats:
    file_count: int = 0
    replay_bundle_count: int = 0


@dataclass(frozen=True, slots=True)
class _TextPayload:
    text: str
    rows: tuple[tuple[int, str], ...]


def run_artifact_hygiene_scan(*, repo_root: Path, artifact_roots: Sequence[Path]) -> ScanSummary:
    repo_findings, repo_stats = scan_tracked_repo_tree(repo_root)
    artifact_findings, artifact_stats = scan_artifact_roots(artifact_roots, repo_root=repo_root)
    findings = tuple(
        sorted(
            (*repo_findings, *artifact_findings),
            key=lambda item: (item.surface, item.path, item.rule),
        )
    )
    return ScanSummary(
        findings=findings,
        repo_file_count=repo_stats.file_count,
        artifact_file_count=artifact_stats.file_count,
        replay_bundle_count=artifact_stats.replay_bundle_count,
    )


def default_artifact_roots(repo_root: Path) -> tuple[Path, ...]:
    runs_dir = repo_root / "runs"
    if runs_dir.is_dir():
        return (runs_dir,)
    return ()


def scan_tracked_repo_tree(repo_root: Path) -> tuple[tuple[Finding, ...], _ScanStats]:
    findings: list[Finding] = []
    file_count = 0
    for relative_path in _git_ls_files(repo_root):
        path = repo_root / relative_path
        if not path.exists():
            continue
        file_count += 1
        findings.extend(
            _scan_path(
                surface="repo",
                path=path,
                repo_root=repo_root,
                is_repo_scan=True,
            )
        )
    return tuple(findings), _ScanStats(file_count=file_count)


def scan_artifact_roots(
    artifact_roots: Sequence[Path],
    *,
    repo_root: Path | None = None,
) -> tuple[tuple[Finding, ...], _ScanStats]:
    findings: list[Finding] = []
    file_count = 0
    replay_bundle_count = 0
    for root in artifact_roots:
        resolved_root = root if root.is_absolute() else (repo_root / root if repo_root is not None else root)
        if not resolved_root.exists():
            continue
        for path in sorted(candidate for candidate in resolved_root.rglob("*") if candidate.is_file()):
            file_count += 1
            path_findings, is_replay_bundle = _scan_path_with_replay_count(
                surface="artifact",
                path=path,
                repo_root=repo_root,
                is_repo_scan=False,
            )
            findings.extend(path_findings)
            replay_bundle_count += int(is_replay_bundle)
    return tuple(findings), _ScanStats(file_count=file_count, replay_bundle_count=replay_bundle_count)


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(finding.format() for finding in findings)


def _scan_path_with_replay_count(
    *,
    surface: str,
    path: Path,
    repo_root: Path | None,
    is_repo_scan: bool,
) -> tuple[tuple[Finding, ...], bool]:
    findings = tuple(_scan_path(surface=surface, path=path, repo_root=repo_root, is_repo_scan=is_repo_scan))
    return findings, _is_replay_bundle(path)


def _scan_path(*, surface: str, path: Path, repo_root: Path | None, is_repo_scan: bool) -> list[Finding]:
    findings: list[Finding] = []
    display_path = _display_path(path, repo_root=repo_root)
    suffix = path.suffix.lower()
    normalized_path = _normalize_path(path)
    should_scan_path = _should_scan_path(path=path, is_repo_scan=is_repo_scan)

    if should_scan_path and suffix in FORBIDDEN_BINARY_SUFFIXES:
        findings.append(
            Finding(
                surface=surface,
                path=display_path,
                rule="forbidden_binary_asset",
                detail=f"forbidden bundled asset type {suffix}",
            )
        )

    if (
        should_scan_path
        and suffix in IMAGE_FILE_SUFFIXES
        and _contains_marker(normalized_path, SUSPICIOUS_IMAGE_PATH_MARKERS)
    ):
        findings.append(
            Finding(
                surface=surface,
                path=display_path,
                rule="suspicious_image_asset",
                detail="likely logo or artwork asset path",
            )
        )

    if (
        should_scan_path
        and suffix in TEXT_FILE_SUFFIXES
        and _contains_marker(normalized_path, SUSPICIOUS_TEXT_PATH_MARKERS)
    ):
        findings.append(
            Finding(
                surface=surface,
                path=display_path,
                rule="suspicious_text_asset",
                detail="likely bundled card-text file path",
            )
        )

    if _should_scan_text(path=path, is_repo_scan=is_repo_scan):
        findings.extend(_scan_text_file(surface=surface, path=path, display_path=display_path))

    if _is_replay_bundle(path):
        findings.extend(_scan_replay_bundle(surface=surface, path=path, display_path=display_path))

    return findings


def _scan_text_file(*, surface: str, path: Path, display_path: str) -> list[Finding]:
    suffix = path.suffix.lower()
    payload = _read_text_payload(path)
    findings = _scan_text_payload(surface=surface, display_path=display_path, payload=payload)
    if suffix == ".csv":
        findings.extend(_scan_csv_headers(surface=surface, path=path, display_path=display_path))
    return findings


def _scan_replay_bundle(*, surface: str, path: Path, display_path: str) -> list[Finding]:
    findings: list[Finding] = []
    with zipfile.ZipFile(path, "r") as archive:
        for member_name in sorted(archive.namelist()):
            if member_name.endswith("/"):
                continue
            member_path = Path(member_name)
            member_display_path = f"{display_path}::{member_name}"
            normalized_member_path = _normalize_path(member_path)
            suffix = member_path.suffix.lower()

            if suffix in FORBIDDEN_BINARY_SUFFIXES:
                findings.append(
                    Finding(
                        surface=surface,
                        path=member_display_path,
                        rule="forbidden_binary_asset",
                        detail=f"forbidden replay member asset type {suffix}",
                    )
                )
                continue

            if suffix in IMAGE_FILE_SUFFIXES and _contains_marker(
                normalized_member_path,
                SUSPICIOUS_IMAGE_PATH_MARKERS,
            ):
                findings.append(
                    Finding(
                        surface=surface,
                        path=member_display_path,
                        rule="suspicious_image_asset",
                        detail="likely logo or artwork asset inside replay bundle",
                    )
                )
                continue

            if suffix not in TEXT_FILE_SUFFIXES:
                continue

            payload = _read_zip_text_payload(archive, member_name)
            findings.extend(_scan_text_payload(surface=surface, display_path=member_display_path, payload=payload))
            if suffix == ".csv":
                findings.extend(_scan_csv_payload(surface=surface, display_path=member_display_path, payload=payload))
    return findings


def _scan_text_payload(*, surface: str, display_path: str, payload: _TextPayload) -> list[Finding]:
    findings: list[Finding] = []

    for line_number, line in payload.rows:
        trademark = _find_trademark_marker(line)
        if trademark is not None:
            findings.append(
                Finding(
                    surface=surface,
                    path=display_path,
                    rule="trademark_marker",
                    detail=f"line {line_number} contains franchise marker {trademark!r}",
                )
            )
            break

    card_text_match = _CARD_TEXT_FIELD_RE.search(payload.text)
    if card_text_match is not None:
        findings.append(
            Finding(
                surface=surface,
                path=display_path,
                rule="card_text_field",
                detail=f"contains likely card-text field {card_text_match.group(1)!r}",
            )
        )
    return findings


def _scan_csv_headers(*, surface: str, path: Path, display_path: str) -> list[Finding]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return _scan_csv_rows(surface=surface, display_path=display_path, rows=rows)


def _scan_csv_payload(*, surface: str, display_path: str, payload: _TextPayload) -> list[Finding]:
    reader = csv.reader(payload.text.splitlines())
    return _scan_csv_rows(surface=surface, display_path=display_path, rows=list(reader))


def _scan_csv_rows(*, surface: str, display_path: str, rows: Sequence[Sequence[str]]) -> list[Finding]:
    if not rows:
        return []
    header = tuple(column.strip() for column in rows[0])
    flagged_indices = [
        index for index, column in enumerate(header) if _normalize_field_name(column) in CARD_TEXT_FIELD_NAMES
    ]
    if not flagged_indices:
        return []
    for row in rows[1:]:
        for index in flagged_indices:
            if index < len(row) and row[index].strip():
                return [
                    Finding(
                        surface=surface,
                        path=display_path,
                        rule="card_text_field",
                        detail=f"CSV column {header[index]!r} contains non-empty card-text content",
                    )
                ]
    return []


def _read_text_payload(path: Path) -> _TextPayload:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _TextPayload(text=text, rows=tuple(enumerate(text.splitlines(), start=1)))


def _read_zip_text_payload(archive: zipfile.ZipFile, member_name: str) -> _TextPayload:
    text = archive.read(member_name).decode("utf-8", errors="replace")
    return _TextPayload(text=text, rows=tuple(enumerate(text.splitlines(), start=1)))


def _should_scan_text(*, path: Path, is_repo_scan: bool) -> bool:
    if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
        return False
    return _should_scan_path(path=path, is_repo_scan=is_repo_scan)


def _should_scan_path(*, path: Path, is_repo_scan: bool) -> bool:
    if not is_repo_scan:
        return True
    return not any(part in REPO_SCAN_SKIP_DIRS for part in path.parts)


def _git_ls_files(repo_root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    return tuple(Path(item.decode("utf-8")) for item in completed.stdout.split(b"\x00") if item)


def _display_path(path: Path, *, repo_root: Path | None) -> str:
    resolved_path = path.resolve()
    if repo_root is None:
        return path.as_posix()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _normalize_path(path: Path) -> str:
    return path.as_posix().lower()


def _normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _contains_marker(value: str, markers: Sequence[str]) -> bool:
    tokens = tuple(re.findall(r"[a-z0-9]+", value.lower()))
    return any(_contains_token_sequence(tokens, _marker_tokens(marker)) for marker in markers)


def _contains_token_sequence(tokens: Sequence[str], marker_tokens: Sequence[str]) -> bool:
    marker_length = len(marker_tokens)
    if marker_length == 0 or marker_length > len(tokens):
        return False
    return any(
        tokens[index : index + marker_length] == tuple(marker_tokens)
        for index in range(len(tokens) - marker_length + 1)
    )


def _marker_tokens(marker: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", marker.lower()))


def _find_trademark_marker(value: str) -> str | None:
    for marker_name, pattern in TRADEMARK_PATTERNS:
        if pattern.search(value):
            return marker_name
    return None


def _is_replay_bundle(path: Path) -> bool:
    return path.suffix.lower() == ".zip" and path.name.startswith("replay_")


_CARD_TEXT_FIELD_RE = re.compile(
    r"(?i)(?:^|[\{\[,\s\"'])"
    r"(card_text|effect_text|ability_text|flavo(?:u)?r_text|card_name|character_name|series_name|franchise_name)"
    r"(?:[\"']?\s*[:=])"
)
