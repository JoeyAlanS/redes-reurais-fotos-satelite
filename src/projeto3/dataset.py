"""Leitura das partições salvas com a mesma ordem de classes para todos os modelos."""

import csv
from pathlib import Path

import tensorflow as tf

from .classes import CLASS_TO_INDEX


def load_partition(root: Path, partition: str) -> tuple[list[str], list[int]]:
    if partition not in {"treino", "validacao", "teste"}:
        raise ValueError("Partição inválida")
    path = root / "dados/splits/particoes.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["particao"] == partition]
    if not rows:
        raise ValueError(f"Partição vazia: {partition}")
    files = [str(root / row["arquivo"]) for row in rows]
    if any(not Path(file).is_file() for file in files):
        raise FileNotFoundError("Algumas imagens listadas nas partições não existem")
    return files, [CLASS_TO_INDEX[row["classe"]] for row in rows]


def make_dataset(root: Path, partition: str, resolution: int, batch_size: int) -> tf.data.Dataset:
    paths, labels = load_partition(root, partition)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if partition == "treino":
        dataset = dataset.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (resolution, resolution))
        return tf.cast(image, tf.float32) / 255.0, label

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
