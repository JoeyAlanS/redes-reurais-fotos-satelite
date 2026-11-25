"""Treina uma execução final por vez, sempre com pesos inicializados do zero.

O checkpoint e a parada antecipada usam o F1 macro de validação (métrica principal de seleção).
Nada do conjunto de teste é lido aqui.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from projeto3.dataset import load_partition, make_dataset  # noqa: E402
from projeto3.experimentos import RUNS, cpu_name, library_versions  # noqa: E402
from projeto3.models import build_model, describe  # noqa: E402

LEARNING_RATE = 1e-3
SEED = 42


class SelecaoF1Macro(tf.keras.callbacks.Callback):
    """Calcula o F1 macro de validação a cada época, guarda o melhor checkpoint e para por paciência."""

    def __init__(self, validation, labels, checkpoint: Path, export_model: tf.keras.Model, patience: int):
        super().__init__()
        self.validation, self.labels = validation, np.asarray(labels)
        self.checkpoint, self.export_model, self.patience = checkpoint, export_model, patience
        self.best_f1, self.best_epoch, self.best_weights, self.wait = -1.0, 0, None, 0
        self.best_logs, self.epoch_times = {}, []

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        predicted = np.argmax(self.model.predict(self.validation, verbose=0), axis=1)
        f1 = float(f1_score(self.labels, predicted, average="macro", zero_division=0))
        logs["val_f1_macro"] = f1
        logs["tempo_epoca_s"] = time.perf_counter() - self.started
        self.epoch_times.append(logs["tempo_epoca_s"])
        if f1 > self.best_f1:
            self.best_f1, self.best_epoch, self.wait = f1, epoch + 1, 0
            self.best_weights = self.model.get_weights()
            self.best_logs = {key: float(value) for key, value in logs.items()}
            # Exporta sem estado do otimizador: o arquivo salvo serve só para inferência.
            self.export_model.set_weights(self.best_weights)
            self.export_model.save(self.checkpoint)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        print(f" — val_f1_macro={f1:.4f} (melhor {self.best_f1:.4f} na época {self.best_epoch})")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def hardware() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    names = [tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name) for gpu in gpus]
    return {"dispositivo": "GPU" if gpus else "CPU", "gpus": names, "cpu": cpu_name(),
            "nucleos_cpu": os.cpu_count(), "sistema": platform.platform()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experimento", choices=RUNS)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    family, variant, resolution = RUNS[args.experimento]
    batch = int(os.getenv(f"BATCH_SIZE_{resolution}", 32))
    epochs = int(os.getenv("MAX_EPOCHS", 40))
    patience = int(os.getenv("PATIENCE", 8))
    augmentation = os.getenv("AUGMENTATION", "1") == "1"
    if batch < 1 or epochs < 1 or patience < 1:
        raise ValueError("BATCH_SIZE, MAX_EPOCHS e PATIENCE devem ser positivos")

    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    train = make_dataset(ROOT, "treino", resolution, batch, augmentation=augmentation)
    validation = make_dataset(ROOT, "validacao", resolution, batch)
    train_files, _ = load_partition(ROOT, "treino")
    validation_files, validation_labels = load_partition(ROOT, "validacao")

    model = build_model(family, variant, resolution)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    summary = describe(model)
    print(summary)

    checkpoint = ROOT / "modelos" / f"{args.experimento}.keras"
    history_path = ROOT / "resultados" / "historicos" / f"{args.experimento}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    selection = SelecaoF1Macro(validation, validation_labels, checkpoint,
                               build_model(family, variant, resolution), patience)
    started = time.perf_counter()
    model.fit(train, validation_data=validation, epochs=epochs, verbose=2,
                        callbacks=[selection, tf.keras.callbacks.CSVLogger(history_path)])
    elapsed = time.perf_counter() - started

    # Confere o arquivo salvo: o F1 recalculado deve coincidir com o da época escolhida.
    saved = tf.keras.models.load_model(checkpoint)
    predicted = np.argmax(saved.predict(validation, verbose=0), axis=1)
    f1_saved = float(f1_score(validation_labels, predicted, average="macro", zero_division=0))
    epoch_times = selection.epoch_times
    metadata = {
        "experimento": args.experimento, "familia": family, "variante": variant, "resolucao": resolution,
        "parametros": int(saved.count_params()),
        "parametros_treinaveis": int(sum(np.prod(w.shape) for w in saved.trainable_weights)),
        "funcao_perda": "sparse_categorical_crossentropy", "otimizador": "Adam",
        "taxa_aprendizagem": LEARNING_RATE, "lote": batch, "limite_epocas": epochs,
        "criterio_parada": f"F1 macro de validação sem melhora por {patience} épocas; restaura a melhor época",
        "paciencia": patience, "epocas_executadas": len(epoch_times), "checkpoint_epoca": selection.best_epoch,
        "f1_macro_validacao_checkpoint": f1_saved,
        "acuracia_validacao_checkpoint": float(accuracy_score(validation_labels, predicted)),
        "perda_validacao_checkpoint": selection.best_logs.get("val_loss"),
        "imagens_treino": len(train_files), "imagens_validacao": len(validation_files),
        "preprocessamento": "RGB, redimensionamento bilinear com antialias a partir do original 256x256, pixels/255",
        "aumento": "espelhamento horizontal/vertical e rotação de 90° aleatórios (só treino)" if augmentation else "nenhum",
        "semente": SEED,
        "tempo_total_segundos": elapsed,
        "tempo_medio_epoca_segundos": float(np.mean(epoch_times)),
        "metodo_tempo": ("time.perf_counter em volta de model.fit; inclui leitura/cache das imagens na 1ª época "
                         "e a predição de validação usada no F1; exclui download e preparação das partições"),
        "tamanho_modelo_bytes": checkpoint.stat().st_size,
        "hardware": hardware(),
        "versoes": library_versions("tensorflow", "keras", "numpy", "scikit-learn", "pillow"),
        "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "modelos" / f"{args.experimento}_resumo.txt").write_text(summary, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
