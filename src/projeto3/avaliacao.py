"""Métricas, tabelas e figuras da Seção 4 a partir dos arquivos de previsão salvos.

Não depende do TensorFlow: as previsões de cada modelo (inclusive o YOLO do bônus) ficam em
resultados/previsoes/<experimento>_<particao>.csv com as probabilidades das dez classes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)

from .classes import CLASSES, NOMES_PT  # noqa: E402

PROB_COLUMNS = [f"p_{label}" for label in CLASSES]
SHORT = [NOMES_PT[label] for label in CLASSES]
FONTE = "Fonte: elaborado pelos autores a partir do NWPU-RESISC45 (Cheng; Han; Lu, 2017)."


def write_predictions(path: Path, files: list[str], labels: list[int], probabilities: np.ndarray, root: Path):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        "arquivo": [Path(f).resolve().relative_to(root.resolve()).as_posix() for f in files],
        "classe_real": [CLASSES[i] for i in labels],
        "classe_prevista": [CLASSES[i] for i in predicted],
        "confianca": probabilities.max(axis=1),
    })
    frame[PROB_COLUMNS] = probabilities
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    return frame


def prediction_path(root: Path, run: str, partition: str) -> Path:
    return root / "resultados" / "previsoes" / f"{run}_{partition}.csv"


def load_predictions(root: Path, run: str, partition: str) -> pd.DataFrame | None:
    path = prediction_path(root, run, partition)
    return pd.read_csv(path) if path.is_file() else None


def load_metadata(root: Path, run: str) -> dict:
    path = root / "modelos" / f"{run}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_history(root: Path, run: str) -> pd.DataFrame | None:
    path = root / "resultados" / "historicos" / f"{run}.csv"
    return pd.read_csv(path) if path.is_file() else None


def compute_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Resumo, relatório por classe (ordem fixa) e matriz de confusão (reais nas linhas)."""
    actual, predicted = frame["classe_real"], frame["classe_prevista"]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=list(CLASSES), zero_division=0)
    predicted_counts = predicted.value_counts()
    # Precisão é 0/0 quando a classe nunca é prevista; sklearn usa zero_division=0 e registramos o caso.
    never_predicted = [label for label in CLASSES if predicted_counts.get(label, 0) == 0]
    report = pd.DataFrame({"classe": CLASSES, "precisao": precision, "recall": recall, "f1": f1, "suporte": support})
    summary = {
        "acuracia": accuracy_score(actual, predicted),
        "f1_macro": f1_score(actual, predicted, labels=list(CLASSES), average="macro", zero_division=0),
        "precisao_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
        "classes_sem_previsao": ";".join(never_predicted),
    }
    matrix = confusion_matrix(actual, predicted, labels=list(CLASSES))
    return summary, report, matrix


def _note(fig):
    fig.text(0.01, 0.003, FONTE, fontsize=7, color="0.35", ha="left", va="bottom")


def plot_history(history: pd.DataFrame, run: str, checkpoint_epoch: int | None, axes):
    epochs = history["epoch"] + 1
    loss_ax, acc_ax = axes
    loss_ax.plot(epochs, history["loss"], label="treino")
    loss_ax.plot(epochs, history["val_loss"], label="validação")
    loss_ax.set_title(f"{run} — perda")
    acc_ax.set_title(f"{run} — acurácia / F1")
    if "accuracy" in history and history["accuracy"].notna().any():
        acc_ax.plot(epochs, history["accuracy"], label="acurácia treino")
    acc_ax.plot(epochs, history["val_accuracy"], label="acurácia validação")
    if "val_f1_macro" in history:
        acc_ax.plot(epochs, history["val_f1_macro"], "--", label="F1 macro validação")
    for ax in axes:
        if checkpoint_epoch:
            ax.axvline(checkpoint_epoch, color="crimson", lw=1, ls=":", label=f"checkpoint (ép. {checkpoint_epoch})")
        ax.set_xlabel("época")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)


def plot_confusion(matrix: np.ndarray, ax, title: str, normalize: bool, labels=SHORT):
    if normalize:
        rows = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        text = [[f"{v:.2f}" if v >= 0.005 else "" for v in row] for row in values]
    else:
        values = matrix
        image = ax.imshow(values, cmap="Blues")
        text = [[str(v) if v else "" for v in row] for row in values]
    limit = values.max() * 0.6
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6.5,
                    color="white" if values[i, j] > limit else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.set_title(title, fontsize=9)
    return image


def plot_errors(root: Path, frame: pd.DataFrame, run: str, path: Path, limit: int = 6) -> pd.DataFrame:
    """Até seis erros de teste, os de maior confiança primeiro (casos em que o modelo errou com convicção)."""
    errors = frame[frame["classe_real"] != frame["classe_prevista"]]
    errors = errors.sort_values(["confianca", "arquivo"], ascending=[False, True]).head(limit)
    if errors.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{run}: nenhum erro no teste", ha="center", va="center")
    else:
        fig, axes = plt.subplots(1, len(errors), figsize=(2.6 * len(errors), 4.0), squeeze=False)
        for ax, (_, row) in zip(axes[0], errors.iterrows()):
            with Image.open(root / row["arquivo"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"real: {NOMES_PT[row['classe_real']]}\nprevista: {NOMES_PT[row['classe_prevista']]}\n"
                         f"confiança: {row['confianca']:.2f}", fontsize=8)
            ax.set_xlabel(Path(row["arquivo"]).name, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Erros de teste de maior confiança — {run}", fontsize=10)
    _note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return errors
