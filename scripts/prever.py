"""Gera as previsões dos modelos salvos (probabilidades das dez classes) sem treinar de novo.

Por padrão só a validação é prevista. O teste exige --teste e deve ser rodado uma única vez,
depois de encerradas todas as escolhas de arquitetura e treino.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tensorflow as tf  # noqa: E402
from projeto3.avaliacao import prediction_path, write_predictions  # noqa: E402
from projeto3.dataset import load_partition, make_dataset  # noqa: E402
from projeto3.experimentos import RUNS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentos", nargs="*", help=f"padrão: todos ({', '.join(RUNS)})")
    parser.add_argument("--teste", action="store_true", help="também prevê o conjunto de teste")
    args = parser.parse_args()
    unknown = set(args.experimentos) - set(RUNS)
    if unknown:
        parser.error(f"experimentos desconhecidos: {', '.join(sorted(unknown))}")
    load_dotenv(ROOT / ".env")
    partitions = ["validacao", "teste"] if args.teste else ["validacao"]
    for run in args.experimentos or RUNS:
        checkpoint = ROOT / "modelos" / f"{run}.keras"
        if not checkpoint.is_file():
            print(f"{run}: modelo ausente em {checkpoint.relative_to(ROOT)}; pulando")
            continue
        model = tf.keras.models.load_model(checkpoint)
        resolution = RUNS[run][2]
        batch = int(os.getenv(f"BATCH_SIZE_{resolution}", 32))
        for partition in partitions:
            files, labels = load_partition(ROOT, partition)
            probabilities = model.predict(make_dataset(ROOT, partition, resolution, batch), verbose=0)
            frame = write_predictions(prediction_path(ROOT, run, partition), files, labels, probabilities, ROOT)
            accuracy = (frame["classe_real"] == frame["classe_prevista"]).mean()
            print(f"{run} [{partition}]: {len(frame)} imagens, acurácia {accuracy:.4f}")


if __name__ == "__main__":
    main()
