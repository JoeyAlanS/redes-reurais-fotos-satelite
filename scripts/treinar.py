"""Treina uma execução final por vez, sempre com pesos inicializados do zero."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from projeto3.dataset import load_partition, make_dataset  # noqa: E402
from projeto3.models import build_model  # noqa: E402

RUNS = {
    "MLP-A-128": ("MLP", "A", 128), "MLP-B-128": ("MLP", "B", 128),
    "CNN-A-128": ("CNN", "A", 128), "CNN-B-128": ("CNN", "B", 128),
    "CNN-A-256": ("CNN", "A", 256), "CNN-B-256": ("CNN", "B", 256),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experimento", choices=RUNS)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    family, variant, resolution = RUNS[args.experimento]
    batch = int(os.getenv(f"BATCH_SIZE_{resolution}", 16 if resolution == 128 else 8))
    epochs = int(os.getenv("MAX_EPOCHS", 12))
    if batch < 1 or epochs < 1:
        raise ValueError("BATCH_SIZE e MAX_EPOCHS devem ser positivos")
    tf.keras.utils.set_random_seed(42)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    train = make_dataset(ROOT, "treino", resolution, batch)
    validation = make_dataset(ROOT, "validacao", resolution, batch)
    model = build_model(family, variant, resolution)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    checkpoint = ROOT / "modelos" / f"{args.experimento}.keras"
    history_path = ROOT / "resultados/historicos" / f"{args.experimento}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(history_path),
    ]
    started = time.monotonic()
    history = model.fit(train, validation_data=validation, epochs=epochs, callbacks=callbacks)
    elapsed = time.monotonic() - started
    saved = tf.keras.models.load_model(checkpoint)
    _, actual = load_partition(ROOT, "validacao")
    predicted = np.argmax(saved.predict(validation, verbose=0), axis=1)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    metadata = {
        "experimento": args.experimento, "parametros": saved.count_params(),
        "funcao_perda": "sparse_categorical_crossentropy", "otimizador": "Adam",
        "taxa_aprendizagem": 0.001, "lote": batch, "limite_epocas": epochs,
        "epocas_executadas": len(history.history["loss"]), "checkpoint_epoca": best_epoch,
        "criterio_parada": "val_loss, paciencia=3", "f1_macro_validacao_checkpoint": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "normalizacao": "RGB, resize a partir do original, pixels/255", "aumento": "nenhum",
        "semente": 42, "tempo_total_segundos": elapsed,
        "tempo_medio_epoca_segundos": elapsed / len(history.history["loss"]),
        "tamanho_modelo_bytes": checkpoint.stat().st_size,
        "tensorflow": tf.__version__, "hardware": [gpu.name for gpu in tf.config.list_physical_devices("GPU")] or ["CPU"],
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
