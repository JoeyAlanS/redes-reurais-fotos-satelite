"""Bônus: ajusta um YOLO pré-treinado no modo classify às mesmas partições.

Roda no ambiente separado .venv-yolo (requirements-yolo.txt). Gera previsões, histórico e
metadados no mesmo formato das seis execuções, para que scripts/avaliar.py inclua o YOLO
nas tabelas, matrizes e na comparação com a CNN de referência.
"""

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from projeto3.avaliacao import prediction_path, write_predictions  # noqa: E402
from projeto3.classes import CLASS_TO_INDEX, CLASSES  # noqa: E402
from projeto3.experimentos import YOLO_RUN, cpu_name, library_versions  # noqa: E402

SPLIT_NAMES = {"treino": "train", "validacao": "val", "teste": "test"}


def read_partitions() -> dict[str, list[tuple[str, str]]]:
    partitions = {name: [] for name in SPLIT_NAMES}
    with (ROOT / "dados/splits/particoes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            partitions[row["particao"]].append((row["arquivo"], row["classe"]))
    return partitions


def build_folder_dataset(partitions, target: Path):
    """Estrutura train/val/test/<classe>/ com links simbólicos para os originais (nada é copiado)."""
    if target.exists():
        shutil.rmtree(target)
    for partition, rows in partitions.items():
        for relative, label in rows:
            link = target / SPLIT_NAMES[partition] / label / Path(relative).name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(ROOT / relative, link.parent))


def predict(model, rows, imgsz: int, batch: int) -> np.ndarray:
    order = [model.names[i] for i in range(len(model.names))]
    to_fixed = [order.index(label) for label in CLASSES]
    probabilities = []
    for start in range(0, len(rows), batch):
        chunk = [str(ROOT / relative) for relative, _ in rows[start:start + batch]]
        for result in model.predict(chunk, imgsz=imgsz, verbose=False):
            probabilities.append(result.probs.data.cpu().numpy()[to_fixed])
    return np.stack(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pesos", default="yolo11n-cls.pt", help="pesos pré-treinados (ImageNet) da Ultralytics")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--lote", type=int, default=32)
    args = parser.parse_args()

    partitions = read_partitions()
    data = ROOT / "dados" / "yolo"
    build_folder_dataset(partitions, data)
    workdir = ROOT / "modelos" / "yolo"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # pesos baixados pela Ultralytics ficam aqui, fora da raiz do repositório

    model = YOLO(args.pesos)
    started = time.perf_counter()
    model.train(data=str(data), epochs=args.epocas, patience=args.paciencia, imgsz=args.imgsz, batch=args.lote,
                seed=42, deterministic=True, project=str(workdir), name=YOLO_RUN, exist_ok=True, plots=True)
    elapsed = time.perf_counter() - started
    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    exported = ROOT / "modelos" / f"{YOLO_RUN}.pt"
    shutil.copy2(best_weights, exported)

    best = YOLO(str(exported))
    validation_f1 = None
    for partition in ("validacao", "teste"):
        rows = partitions[partition]
        probabilities = predict(best, rows, args.imgsz, args.lote)
        labels = [CLASS_TO_INDEX[label] for _, label in rows]
        write_predictions(prediction_path(ROOT, YOLO_RUN, partition), [str(ROOT / r) for r, _ in rows],
                          labels, probabilities, ROOT)
        if partition == "validacao":
            predicted = probabilities.argmax(axis=1)
            validation_f1 = float(f1_score(labels, predicted, average="macro", zero_division=0))
            validation_accuracy = float(accuracy_score(labels, predicted))

    # Histórico no mesmo formato do Keras; a Ultralytics não registra acurácia de treino.
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = [c.strip() for c in results.columns]
    history = pd.DataFrame({
        "epoch": results["epoch"] - results["epoch"].min(),
        "loss": results["train/loss"], "val_loss": results["val/loss"],
        "accuracy": np.nan, "val_accuracy": results["metrics/accuracy_top1"],
    })
    history_path = ROOT / "resultados" / "historicos" / f"{YOLO_RUN}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    train_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(model.trainer.args).items()}
    metadata = {
        "experimento": YOLO_RUN, "familia": "YOLO", "variante": Path(args.pesos).stem, "resolucao": args.imgsz,
        "parametros": int(sum(p.numel() for p in best.model.parameters())),
        "origem_pesos": f"Ultralytics {args.pesos}, pré-treinado no ImageNet (download automático da Ultralytics)",
        "lote": args.lote, "limite_epocas": args.epocas, "paciencia": args.paciencia,
        "epocas_executadas": len(history),
        "checkpoint_epoca": int(history["val_accuracy"].idxmax()) + 1,
        "criterio_checkpoint": "padrão Ultralytics: maior acurácia top-1 de validação (fitness)",
        "f1_macro_validacao_checkpoint": validation_f1, "acuracia_validacao_checkpoint": validation_accuracy,
        "tempo_total_segundos": elapsed, "tempo_medio_epoca_segundos": elapsed / len(history),
        "metodo_tempo": "time.perf_counter em volta de model.train (inclui validação por época); exclui download dos pesos",
        "tamanho_modelo_bytes": exported.stat().st_size,
        "argumentos_treino": train_args,
        "hardware": {"dispositivo": "GPU" if gpu else "CPU", "gpus": [gpu] if gpu else [], "cpu": cpu_name(),
                     "nucleos_cpu": os.cpu_count(), "sistema": platform.platform()},
        "versoes": library_versions("ultralytics", "torch", "torchvision", "numpy", "scikit-learn"),
        "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ROOT / "modelos" / f"{YOLO_RUN}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "argumentos_treino"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
