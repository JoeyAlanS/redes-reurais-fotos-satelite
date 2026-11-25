"""Leitura das partições salvas com a mesma ordem de classes para todos os modelos."""

import csv
from pathlib import Path

import tensorflow as tf

from .classes import CLASS_TO_INDEX

SEED = 42


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


def augment(image: tf.Tensor) -> tf.Tensor:
    """Cenas aéreas não têm orientação canônica: espelhamentos e giros de 90° preservam a classe."""
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    return tf.image.rot90(image, tf.random.uniform((), 0, 4, dtype=tf.int32, seed=SEED))


def make_dataset(root: Path, partition: str, resolution: int, batch_size: int,
                 augmentation: bool = False) -> tf.data.Dataset:
    """RGB decodificado do original, redimensionado e dividido por 255; aumento só no treino."""
    if augmentation and partition != "treino":
        raise ValueError("Aumento de dados atua apenas no treino")
    paths, labels = load_partition(root, partition)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        return image, label

    def resize(image, label):
        image = tf.image.resize(image, (resolution, resolution), antialias=True)
        image = tf.clip_by_value(image, 0.0, 255.0) / 255.0
        if augmentation:
            image = augment(image)
        return image, label

    # Cache dos originais em uint8 (~1 GB no treino); o redimensionamento parte sempre do original.
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if partition == "treino":
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
