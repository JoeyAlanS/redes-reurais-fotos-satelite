"""Inventário e partições persistidas sem modificar imagens originais."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .classes import CLASSES

SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)


def dhash_variants(image: Image.Image) -> np.ndarray:
    """dHash de 256 bits nas oito orientações (rotações de 90° e espelhamentos).

    Retorna (8, 4) uint64; a linha 0 é a orientação original. Comparar a original de uma
    imagem com as oito variantes da outra detecta cópias giradas ou espelhadas.
    """
    pixels = np.asarray(image.convert("L").resize((17, 17), Image.Resampling.LANCZOS), dtype=np.float32)
    variants = []
    for flipped in (pixels, np.fliplr(pixels)):
        for turns in range(4):
            grid = np.rot90(flipped, turns)
            bits = grid[:16, 1:] > grid[:16, :-1]
            variants.append(np.packbits(bits.ravel()).view(np.uint64))
    return np.stack(variants)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 100) -> list[tuple[int, int, int]]:
    """Pares (i, j, distância) com i < j e distância de Hamming mínima entre orientações <= limiar."""
    original = hashes[:, 0, :]
    pairs = {}
    for start in range(0, len(hashes), chunk):
        block = original[start:start + chunk, None, None, :] ^ hashes[None, :, :, :]
        distances = np.bitwise_count(block).sum(axis=-1).min(axis=-1)
        rows, cols = np.nonzero(distances <= threshold)
        for row, col in zip(rows, cols):
            i, j = start + int(row), int(col)
            if i != j:
                # A distância não é simétrica (original de i contra variantes de j); fica a menor.
                key = (min(i, j), max(i, j))
                pairs[key] = min(pairs.get(key, threshold + 1), int(distances[row, col]))
    return [(i, j, distance) for (i, j), distance in sorted(pairs.items())]


def _union_find_groups(size: int, pairs) -> list[list[int]]:
    parent = list(range(size))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j, *_ in pairs:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)
    groups = defaultdict(list)
    for node in range(size):
        groups[find(node)].append(node)
    return list(groups.values())


def prepare(data_dir: Path, root: Path, threshold: int = 12) -> dict:
    root = root.resolve()
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset ausente: {data_dir}")
    rows, hashes, issues = [], [], []
    sizes, modes = Counter(), Counter()
    for label in CLASSES:
        folder = data_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Classe ausente: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file())
        for path in files:
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                issues.append({"arquivo": relative, "problema": "extensão inesperada"})
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    size, mode = image.size, image.mode
                    hashes.append(dhash_variants(image))
            except (OSError, UnidentifiedImageError) as error:
                issues.append({"arquivo": relative, "problema": f"leitura: {error}"})
                continue
            sizes[f"{size[0]}x{size[1]}"] += 1
            modes[mode] += 1
            if size != (256, 256) or mode != "RGB":
                issues.append({"arquivo": relative, "problema": f"dimensão={size}; canais={mode}"})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"arquivo": relative, "classe": label, "sha256": digest})
    near_pairs = near_duplicate_pairs(np.stack(hashes), threshold)
    origin = data_dir.relative_to(root).as_posix() if data_dir.is_relative_to(root) else data_dir.name
    return partition_and_save(rows, near_pairs, root, origin, threshold, dict(sizes), dict(modes), issues)


def partition_and_save(rows: list[dict], near_pairs, root: Path, origin: str, threshold: int,
                       sizes: dict, modes: dict, issues: list) -> dict:
    """Agrupa duplicatas, sorteia as partições e grava particoes.csv, quase_duplicatas.csv e inventario.json.

    Depende só dos caminhos, classes, SHA-256 e pares de quase-duplicatas, não das imagens.
    """
    # Duplicatas exatas (mesmo SHA-256) e versões semelhantes (dHash) formam grupos indivisíveis.
    by_hash = defaultdict(list)
    for index, row in enumerate(rows):
        by_hash[row["sha256"]].append(index)
    exact_pairs = [(g[0], other) for g in by_hash.values() for other in g[1:]]
    groups = _union_find_groups(len(rows), exact_pairs + near_pairs)
    group_of = {}
    for number, members in enumerate(groups):
        for member in members:
            group_of[member] = number
    for index, row in enumerate(rows):
        row["grupo"] = group_of[index]

    # Sorteio estratificado por classe sobre grupos; um grupo segue a classe do primeiro arquivo
    # em ordem alfabética e leva todos os membros (inclusive de outra classe) para a mesma partição.
    rng = random.Random(SEED)
    for label in CLASSES:
        owned = [g for g in groups if rows[min(g, key=lambda i: rows[i]["arquivo"])]["classe"] == label]
        owned = [sorted(g, key=lambda i: rows[i]["arquivo"]) for g in owned]
        owned.sort(key=lambda g: rows[g[0]]["arquivo"])
        rng.shuffle(owned)
        total = sum(len(g) for g in owned)
        targets = (round(total * FRACTIONS[0]), round(total * (FRACTIONS[0] + FRACTIONS[1])))
        count = 0
        for group in owned:
            partition = "treino" if count < targets[0] else "validacao" if count < targets[1] else "teste"
            for member in group:
                rows[member]["particao"] = partition
            count += len(group)

    output = root / "dados" / "splits"
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["arquivo"])
    with (output / "particoes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("arquivo", "classe", "particao", "grupo", "sha256"))
        writer.writeheader()
        writer.writerows(ordered)
    with (output / "quase_duplicatas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("arquivo_a", "classe_a", "arquivo_b", "classe_b", "distancia_hamming", "duplicata_exata"))
        exact = {(min(i, j), max(i, j)) for i, j in exact_pairs}
        merged = {(min(i, j), max(i, j)): d for i, j, d in near_pairs}
        merged.update({pair: 0 for pair in exact if pair not in merged})
        for (i, j), distance in sorted(merged.items(), key=lambda item: (rows[item[0][0]]["arquivo"], rows[item[0][1]]["arquivo"])):
            writer.writerow((rows[i]["arquivo"], rows[i]["classe"], rows[j]["arquivo"], rows[j]["classe"],
                             distance, rows[i]["sha256"] == rows[j]["sha256"]))

    multi = [g for g in groups if len(g) > 1]
    cross = [sorted(rows[i]["arquivo"] for i in g) for g in multi if len({rows[i]["classe"] for i in g}) > 1]
    leaks = [g for g in multi if len({rows[i]["particao"] for i in g}) > 1]
    if leaks:
        raise AssertionError("Grupo de duplicatas dividido entre partições")
    report = {
        "origem": origin,
        "semente": SEED, "proporcoes": dict(zip(("treino", "validacao", "teste"), FRACTIONS)), "total": len(rows),
        "contagens": {label: {p: sum(1 for r in rows if r["classe"] == label and r["particao"] == p)
                              for p in ("treino", "validacao", "teste")} for label in CLASSES},
        "contagens_particao": dict(Counter(row["particao"] for row in rows)),
        "dimensoes": sizes, "modos_cor": modes,
        "duplicatas_exatas": sum(len(g) - 1 for g in by_hash.values()),
        "grupos_exatos": [sorted(rows[i]["arquivo"] for i in g) for g in by_hash.values() if len(g) > 1],
        "quase_duplicatas": {
            "metodo": "dHash 256 bits (17x17 em cinza), mínimo nas 8 orientações, distância de Hamming",
            "limiar": threshold, "pares": len(near_pairs),
            "grupos_com_mais_de_um_arquivo": len(multi),
            "imagens_em_grupos": sum(len(g) for g in multi),
            "maior_grupo": max((len(g) for g in groups), default=1),
            "grupos_entre_classes": cross,
        },
        "exclusoes": [],
        "problemas": issues,
        "regra": ("Grupos de duplicatas exatas e quase-duplicatas são sorteados como unidade, com semente 42, "
                  "após ordenar os caminhos; as proporções por classe podem variar em poucas imagens."),
    }
    (output / "inventario.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
