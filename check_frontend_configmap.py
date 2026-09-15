#!/usr/bin/env python3
"""Keep k8s/frontend-configmap.yaml in sync with the files in k8s/frontend/.

The ConfigMap embeds a verbatim copy of every file in k8s/frontend/. Nothing
stops the two from drifting, and when they do the symptom is a frontend that
silently serves stale JS -- so this is checked in CI.

  python3 scripts/check_frontend_configmap.py           # verify, exit 1 on drift
  python3 scripts/check_frontend_configmap.py --write   # regenerate from disk

Equivalent to `kubectl create configmap frontend-static --from-file=k8s/frontend
--dry-run=client -o yaml`, minus the kubectl dependency.
"""

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "k8s" / "frontend"
CONFIGMAP = ROOT / "k8s" / "frontend-configmap.yaml"
NAME = "frontend-static"


def _block_style(dumper, data):
    """Render multi-line values as block scalars so the file stays readable."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _block_style, Dumper=yaml.SafeDumper)


def data_from_disk() -> dict[str, str]:
    if not SRC.is_dir():
        sys.exit(f"source directory not found: {SRC}")
    return {p.name: p.read_text() for p in sorted(SRC.iterdir()) if p.is_file()}


def normalize(data: dict[str, str]) -> dict[str, str]:
    # YAML block scalars don't round-trip a trailing newline reliably, and it
    # makes no difference to the served bytes, so compare without it.
    return {k: v.rstrip("\n") for k, v in data.items()}


def write(disk: dict[str, str]) -> None:
    doc = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": NAME},
        "data": disk,
    }
    CONFIGMAP.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=10**6)
    )
    print(f"wrote {CONFIGMAP.relative_to(ROOT)} ({len(disk)} files)")


def check(disk: dict[str, str]) -> int:
    if not CONFIGMAP.exists():
        print(f"missing {CONFIGMAP.relative_to(ROOT)}", file=sys.stderr)
        return 1

    doc = yaml.safe_load(CONFIGMAP.read_text()) or {}
    embedded = normalize(doc.get("data") or {})
    expected = normalize(disk)

    problems = []
    for name in sorted(set(expected) - set(embedded)):
        problems.append(f"  missing from ConfigMap: {name}")
    for name in sorted(set(embedded) - set(expected)):
        problems.append(f"  in ConfigMap but not on disk: {name}")
    for name in sorted(set(embedded) & set(expected)):
        if embedded[name] != expected[name]:
            problems.append(f"  out of date: {name}")

    if problems:
        print("frontend ConfigMap has drifted from k8s/frontend/:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print(
            "\nregenerate with: python3 scripts/check_frontend_configmap.py --write",
            file=sys.stderr,
        )
        return 1

    print(f"frontend ConfigMap is in sync ({len(expected)} files)")
    return 0


if __name__ == "__main__":
    disk = data_from_disk()
    if "--write" in sys.argv[1:]:
        write(disk)
        sys.exit(0)
    sys.exit(check(disk))