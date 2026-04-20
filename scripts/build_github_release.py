from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = REPO_ROOT / "dist"
STAGE_ROOT = DIST_ROOT / "gh_repo" / "TRACER"
RELEASE_ASSET_ROOT = DIST_ROOT / "gh_assets"

ROOT_FILES = [
    ".gitignore",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "environment.cpu.yml",
    "environment.gpu.yml",
    "CODE_AND_DATA_AVAILABILITY.md",
    "THIRD_PARTY_NOTICES.md",
    "data/README.md",
    "outputs/README.md",
]

DIRECTORY_SPECS = [
    {"src": "campaign_mem"},
    {"src": "configs"},
    {"src": "docs"},
    {"src": "figures"},
    {"src": "outputs/results"},
    {"src": "outputs/expert_evidence_annotation_packet"},
    {"src": "scripts"},
    {"src": "data/atlasv2_public"},
    {"src": "data/atlasv2_lopo_family"},
    {"src": "data/atlasv2_workbook"},
    {"src": "data/atlas_raw_public"},
    {
        "src": "data/ait_ads_public",
        "exclude_files": {"ait_ads_canonical_events.csv", "ait_ads_canonical_events.zip"},
    },
    {"src": "data/reference_labels"},
    {"src": "data/cross_dataset_transfer"},
    {"src": "data/examples"},
    {"src": "data/splunk_attack_data_public_probe"},
    {"src": "data/synthetic_cam_lds"},
    {"src": "data/synthetic_cam_lds_controlled"},
]

OVERSIZED_RELEASE_ASSETS = [
    "data/ait_ads_public/ait_ads_canonical_events.csv",
]

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}

IGNORED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(os_path(path))
    path.mkdir(parents=True, exist_ok=True)


def os_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def should_skip_file(path: Path, allowed_suffixes: set[str] | None, exclude_names: set[str]) -> bool:
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    if path.name in exclude_names:
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
        return True
    return False


def copy_tree(
    src_rel: str,
    dst_root: Path,
    *,
    allowed_suffixes: set[str] | None = None,
    exclude_names: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> list[dict[str, object]]:
    src_root = REPO_ROOT / src_rel
    copied: list[dict[str, object]] = []
    exclude_names = exclude_names or set()
    exclude_files = exclude_files or set()
    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        if src_file.name in exclude_files:
            continue
        if should_skip_file(src_file, allowed_suffixes, exclude_names):
            continue
        rel_from_repo = src_file.relative_to(REPO_ROOT)
        dst_file = dst_root / rel_from_repo
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os_path(src_file), os_path(dst_file))
        copied.append(
            {
                "path": rel_from_repo.as_posix(),
                "size_bytes": src_file.stat().st_size,
            }
        )
    return copied


def copy_root_files(dst_root: Path) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for rel in ROOT_FILES:
        src = REPO_ROOT / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os_path(src), os_path(dst))
        copied.append({"path": rel, "size_bytes": src.stat().st_size})
    return copied


def zip_tree(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def zip_single_file(source_file: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(source_file, source_file.name)


def build_release() -> dict[str, object]:
    ensure_clean_dir(STAGE_ROOT)
    RELEASE_ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    copied_files: list[dict[str, object]] = []
    copied_files.extend(copy_root_files(STAGE_ROOT))
    for spec in DIRECTORY_SPECS:
        copied_files.extend(
            copy_tree(
                spec["src"],
                STAGE_ROOT,
                allowed_suffixes=spec.get("allowed_suffixes"),
                exclude_names=spec.get("exclude_names"),
                exclude_files=spec.get("exclude_files"),
            )
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    repo_zip = DIST_ROOT / f"TRACER_github_repo_{timestamp}.zip"
    if repo_zip.exists():
        repo_zip.unlink()
    zip_tree(STAGE_ROOT, repo_zip)

    release_assets = []
    for rel in OVERSIZED_RELEASE_ASSETS:
        src = REPO_ROOT / rel
        asset_zip = RELEASE_ASSET_ROOT / f"{src.stem}.zip"
        if asset_zip.exists():
            asset_zip.unlink()
        zip_single_file(src, asset_zip)
        release_assets.append(
            {
                "source": rel,
                "archive": asset_zip.relative_to(REPO_ROOT).as_posix(),
                "source_size_mb": round(src.stat().st_size / (1024 * 1024), 2),
                "archive_size_mb": round(asset_zip.stat().st_size / (1024 * 1024), 2),
                "sha256": sha256(asset_zip),
            }
        )

    stage_files = sorted(STAGE_ROOT.rglob("*"))
    stage_file_records = []
    for path in stage_files:
        if path.is_file():
            stage_file_records.append(
                {
                    "path": path.relative_to(STAGE_ROOT).as_posix(),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
            )

    largest_stage_files = sorted(stage_file_records, key=lambda item: item["size_mb"], reverse=True)[:25]
    over_100mb = [item for item in stage_file_records if item["size_mb"] > 100.0]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage_root": STAGE_ROOT.relative_to(REPO_ROOT).as_posix(),
        "repo_zip": repo_zip.relative_to(REPO_ROOT).as_posix(),
        "repo_zip_size_mb": round(repo_zip.stat().st_size / (1024 * 1024), 2),
        "repo_zip_sha256": sha256(repo_zip),
        "release_assets": release_assets,
        "copied_file_count": len(stage_file_records),
        "copied_total_size_mb": round(sum(item["size_mb"] for item in stage_file_records), 2),
        "largest_stage_files": largest_stage_files,
        "files_over_100mb_in_stage": over_100mb,
        "excluded_from_git_repo": [
            "external_sources/",
            ".codex/",
            "aris-local-workspace/",
            "Auto-claude-code-research-in-sleep-main/",
            "refine-logs/",
            "_qa_focus/",
            "_qa_focus2/",
            "_qa_pages/",
            "paper/",
            "data/ait_ads_public/ait_ads_canonical_events.csv",
            "local review notes and scratch files",
        ],
    }

    manifest_path = DIST_ROOT / "TRACER_github_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    checksum_lines = [
        f"{manifest['repo_zip_sha256']}  {repo_zip.name}",
        *[f"{asset['sha256']}  {Path(asset['archive']).name}" for asset in release_assets],
    ]
    (DIST_ROOT / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    upload_notes = [
        "# GitHub Upload Instructions",
        "",
        "1. Push the contents of the staged repository tree:",
        f"   - `{manifest['stage_root']}`",
        "2. Or upload the repository snapshot zip:",
        f"   - `{manifest['repo_zip']}`",
        "3. Attach the following oversized-data archive to the GitHub Release page:",
    ]
    for asset in release_assets:
        upload_notes.append(f"   - `{asset['archive']}` ({asset['archive_size_mb']} MB)")
    upload_notes.extend(
        [
            "",
            "Notes:",
            "- The staged repository tree contains no file larger than 100 MB.",
            "- The large AIT-ADS canonical-events table is intentionally shipped as a release asset.",
        ]
    )
    (DIST_ROOT / "UPLOAD_TO_GITHUB.md").write_text("\n".join(upload_notes) + "\n", encoding="utf-8")

    return manifest


def main() -> None:
    manifest = build_release()
    print("GitHub release package created.")
    print(f"Stage directory: {manifest['stage_root']}")
    print(f"Repository zip: {manifest['repo_zip']} ({manifest['repo_zip_size_mb']} MB)")
    print(f"Files over 100 MB in staged repo: {len(manifest['files_over_100mb_in_stage'])}")
    for asset in manifest["release_assets"]:
        print(
            "Release asset: "
            f"{asset['archive']} ({asset['archive_size_mb']} MB, source {asset['source_size_mb']} MB)"
        )


if __name__ == "__main__":
    main()
