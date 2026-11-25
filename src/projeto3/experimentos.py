"""Identificação das seis execuções finais e registro do ambiente de cada uma."""

import platform
import sys
from importlib import metadata
from pathlib import Path

RUNS = {
    "MLP-A-128": ("MLP", "A", 128), "MLP-B-128": ("MLP", "B", 128),
    "CNN-A-128": ("CNN", "A", 128), "CNN-B-128": ("CNN", "B", 128),
    "CNN-A-256": ("CNN", "A", 256), "CNN-B-256": ("CNN", "B", 256),
}
YOLO_RUN = "YOLO-cls"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def library_versions(*packages: str) -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions
