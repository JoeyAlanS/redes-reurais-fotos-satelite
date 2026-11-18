"""Inventário e partições persistidas sem modificar imagens originais."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .classes import CLASSES


def prepare(data_dir: Path, root: Path) -> dict:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset ausente: {data_dir}")
    rows = []
    issues = []
    for label in CLASSES:
        folder = data_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Classe ausente: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file())
        for path in files:
            relative = path.relative_to(root.resolve()).as_posix()
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                issues.append({"arquivo": relative, "problema": "extensão inesperada"})
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    size, mode = image.size, image.mode
            except (OSError, UnidentifiedImageError) as error:
                issues.append({"arquivo": relative, "problema": f"leitura: {error}"})
                continue
            if size != (256, 256) or mode != "RGB":
                issues.append({"arquivo": relative, "problema": f"dimensão={size}; canais={mode}"})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"arquivo": relative, "classe": label, "sha256": digest})

    # Agrupamento de duplicatas exatas antes do sorteio; classes diferentes são erro de rótulo.
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    conflicts = [group for group in by_hash.values() if len({r["classe"] for r in group}) > 1]
    if conflicts:
        raise ValueError(f"{len(conflicts)} grupos de imagens idênticas com classes conflitantes")

    rng = random.Random(42)
    for label in CLASSES:
        groups = [group for group in by_hash.values() if group[0]["classe"] == label]
        groups.sort(key=lambda group: group[0]["arquivo"])
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        targets = (round(total * .70), round(total * .85))
        count = 0
        for group in groups:
            partition = "treino" if count < targets[0] else "validacao" if count < targets[1] else "teste"
            for row in group:
                row["particao"] = partition
            count += len(group)

    rows.sort(key=lambda row: row["arquivo"])
    output = root / "dados" / "splits"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "particoes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("arquivo", "classe", "particao", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "origem": str(data_dir), "semente": 42, "total": len(rows),
        "contagens": {label: dict(Counter(row["particao"] for row in rows if row["classe"] == label)) for label in CLASSES},
        "duplicatas_exatas": sum(len(group) - 1 for group in by_hash.values()),
        "problemas": issues,
        "observacao": "Hashes detectam duplicatas exatas; versões visualmente semelhantes exigem revisão manual antes do treino.",
    }
    (output / "inventario.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
