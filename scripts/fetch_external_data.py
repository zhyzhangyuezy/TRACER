from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]

AIT_ADS_ZIP_URL = "https://zenodo.org/record/8263181/files/ait_ads.zip"
AIT_ADS_REPO_URL = "https://github.com/ait-aecid/alert-data-set.git"
ATLAS_REPO_URL = "https://github.com/purseclab/ATLAS.git"
AIT_ADS_CANONICAL_RELEASE_URL = (
    "https://github.com/zhyzhangyuezy/TRACER/releases/download/v1.0.0/ait_ads_canonical_events.zip"
)
SPLUNK_MEDIA_ROOT = "https://media.githubusercontent.com/media/splunk/attack_data/master"


@dataclass(frozen=True)
class SplunkLogSpec:
    path: str


SPLUNK_CURATED_LOGS: tuple[SplunkLogSpec, ...] = (
    SplunkLogSpec("datasets/attack_techniques/T1046/nmap/horizontal.log"),
    SplunkLogSpec("datasets/attack_techniques/T1046/nmap/vertical.log"),
    SplunkLogSpec("datasets/attack_techniques/T1046/advanced_ip_port_scanner/advanced_ip_port_scanner.log"),
    SplunkLogSpec("datasets/attack_techniques/T1087/enumerate_users_local_group_using_telegram/windows-xml.log"),
    SplunkLogSpec("datasets/attack_techniques/T1059.001/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1059.001/encoded_powershell/explorer_spawns_windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1059.003/cmd_spawns_cscript/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1547.001/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1562.001/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1003.001/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1003.001/atomic_red_team/windows-sysmon_creddump.log"),
    SplunkLogSpec("datasets/attack_techniques/T1003.003/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1021.002/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1021.002/atomic_red_team/smbexec_windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1021.002/atomic_red_team/wmiexec_windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1021.006/lateral_movement/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1021.006/lateral_movement_psh/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon_curl.log"),
    SplunkLogSpec("datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon_curl_upload.log"),
    SplunkLogSpec("datasets/attack_techniques/T1567/gdrive/gdrive_windows.log"),
    SplunkLogSpec("datasets/attack_techniques/T1486/dcrypt/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1486/sam_sam_note/windows-sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1068/windows_escalation_behavior/windows_escalation_behavior_sysmon.log"),
    SplunkLogSpec("datasets/attack_techniques/T1068/drivers/sysmon_sys_filemod.log"),
)

INCLUDED_LABELS = {
    "ait_ads": ROOT / "data" / "reference_labels" / "ait_ads_labels.csv",
    "atlasv2": ROOT / "data" / "reference_labels" / "atlasv2_labels.csv",
}

MATERIALIZED_LABELS = {
    "ait_ads": ROOT / "external_sources" / "AIT-ADS" / "labels.csv",
    "atlasv2": ROOT / "external_sources" / "reapr-ground-truth" / "atlasv2" / "atlasv2_labels.csv",
}


def git_executable() -> str:
    candidates = [
        shutil.which("git"),
        r"C:\Program Files\Git\cmd\git.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("Git executable not found. Install Git before fetching cloned upstream sources.")


def download_file(url: str, destination: Path, *, overwrite: bool) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return {
            "path": str(destination.relative_to(ROOT)),
            "url": url,
            "action": "kept",
            "bytes": int(destination.stat().st_size),
        }
    request = urllib.request.Request(url, headers={"User-Agent": "TRACER-public-release"})
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())
    return {
        "path": str(destination.relative_to(ROOT)),
        "url": url,
        "action": "downloaded" if overwrite or not destination.exists() else "updated",
        "bytes": int(destination.stat().st_size),
    }


def clone_repo(url: str, destination: Path, *, overwrite: bool) -> dict[str, object]:
    if destination.exists():
        if (destination / ".git").exists() and not overwrite:
            return {
                "path": str(destination.relative_to(ROOT)),
                "url": url,
                "action": "kept",
            }
        if overwrite:
            shutil.rmtree(destination)
        else:
            raise RuntimeError(
                f"Destination {destination} exists but is not a git clone. "
                "Use --overwrite to replace it."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [git_executable(), "clone", "--depth", "1", url, str(destination)],
        check=True,
        cwd=ROOT,
    )
    return {
        "path": str(destination.relative_to(ROOT)),
        "url": url,
        "action": "cloned",
    }


def unzip_json_archive(zip_path: Path, output_dir: Path, *, overwrite: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    kept = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not member.filename.lower().endswith(".json"):
                continue
            destination = output_dir / Path(member.filename).name
            if destination.exists() and not overwrite:
                kept += 1
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    return {
        "zip_path": str(zip_path.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "json_files_extracted": extracted,
        "json_files_kept": kept,
    }


def unzip_single_csv(zip_path: Path, output_path: Path, *, overwrite: bool) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return {
            "path": str(output_path.relative_to(ROOT)),
            "action": "kept",
        }
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [member for member in archive.infolist() if member.filename.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"Expected exactly one CSV in {zip_path}, found {len(csv_members)}")
        with archive.open(csv_members[0]) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return {
        "path": str(output_path.relative_to(ROOT)),
        "action": "extracted",
        "bytes": int(output_path.stat().st_size),
    }


def materialize_reference_labels(*, overwrite: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key, source in INCLUDED_LABELS.items():
        destination = MATERIALIZED_LABELS[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            action = "kept"
        else:
            shutil.copy2(source, destination)
            action = "copied"
        records.append(
            {
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "action": action,
                "bytes": int(destination.stat().st_size),
            }
        )
    return records


def fetch_ait_ads_source(*, overwrite: bool) -> dict[str, object]:
    zip_path = ROOT / "external_sources" / "AIT-ADS" / "ait_ads.zip"
    repo_path = ROOT / "external_sources" / "AIT-ADS" / "source_repo"
    unzipped_dir = ROOT / "external_sources" / "AIT-ADS" / "unzipped"
    download_record = download_file(AIT_ADS_ZIP_URL, zip_path, overwrite=overwrite)
    unzip_record = unzip_json_archive(zip_path, unzipped_dir, overwrite=overwrite)
    clone_record = clone_repo(AIT_ADS_REPO_URL, repo_path, overwrite=overwrite)
    return {
        "target": "ait-ads-source",
        "download": download_record,
        "unzip": unzip_record,
        "repo": clone_record,
    }


def fetch_ait_ads_canonical(*, overwrite: bool) -> dict[str, object]:
    asset_zip = ROOT / "data" / "ait_ads_public" / "ait_ads_canonical_events.zip"
    csv_path = ROOT / "data" / "ait_ads_public" / "ait_ads_canonical_events.csv"
    download_record = download_file(AIT_ADS_CANONICAL_RELEASE_URL, asset_zip, overwrite=overwrite)
    extract_record = unzip_single_csv(asset_zip, csv_path, overwrite=overwrite)
    return {
        "target": "ait-ads-canonical",
        "download": download_record,
        "extract": extract_record,
    }


def fetch_atlas_source(*, overwrite: bool) -> dict[str, object]:
    destination = ROOT / "external_sources" / "ATLAS"
    return {
        "target": "atlas-source",
        "repo": clone_repo(ATLAS_REPO_URL, destination, overwrite=overwrite),
    }


def fetch_splunk_probe_raw(*, overwrite: bool) -> dict[str, object]:
    raw_root = ROOT / "external_sources" / "splunk_attack_data_probe" / "raw"
    records = []
    for spec in SPLUNK_CURATED_LOGS:
        destination = raw_root / spec.path
        records.append(download_file(f"{SPLUNK_MEDIA_ROOT}/{spec.path}", destination, overwrite=overwrite))
    return {
        "target": "splunk-probe-raw",
        "downloaded_files": len(records),
        "files": records,
    }


def fetch_reference_labels(*, overwrite: bool) -> dict[str, object]:
    return {
        "target": "reference-labels",
        "files": materialize_reference_labels(overwrite=overwrite),
    }


TARGET_HANDLERS: dict[str, Callable[..., dict[str, object]]] = {
    "reference-labels": fetch_reference_labels,
    "ait-ads-source": fetch_ait_ads_source,
    "ait-ads-canonical": fetch_ait_ads_canonical,
    "atlas-source": fetch_atlas_source,
    "splunk-probe-raw": fetch_splunk_probe_raw,
}


def run_targets(targets: list[str], *, overwrite: bool) -> dict[str, object]:
    selected = list(TARGET_HANDLERS) if "all" in targets else targets
    results = []
    for target in selected:
        if target not in TARGET_HANDLERS:
            raise ValueError(f"Unknown target: {target}")
        results.append(TARGET_HANDLERS[target](overwrite=overwrite))
    return {
        "targets": selected,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch publicly available upstream data and reference assets that are "
            "not fully embedded in the Git-tracked TRACER release."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        choices=["all", *TARGET_HANDLERS.keys()],
        help="One or more fetch targets. Defaults to all publicly scriptable targets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload or replace existing files and clones.",
    )
    parser.add_argument(
        "--status-json",
        default="outputs/results/external_data_fetch_status.json",
        help="Where to store the fetch summary JSON.",
    )
    args = parser.parse_args()

    summary = run_targets(args.targets, overwrite=args.overwrite)
    status_path = ROOT / args.status_json
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("External data setup complete.")
    for item in summary["results"]:
        print(f"- {item['target']}")
    print(f"Status JSON: {status_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
