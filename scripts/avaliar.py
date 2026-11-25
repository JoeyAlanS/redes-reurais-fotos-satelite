"""Consolida métricas, custo, curvas, matrizes de confusão e erros a partir das previsões salvas.

Usa apenas arquivos já gerados (modelos/*.json, resultados/historicos, resultados/previsoes);
não treina nem carrega redes. Avalia o teste somente se as previsões de teste existirem.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402
from projeto3.avaliacao import (SHORT, _note, compute_metrics, load_history, load_metadata,  # noqa: E402
                                load_predictions, plot_confusion, plot_errors, plot_history)
from projeto3.classes import CLASSES  # noqa: E402
from projeto3.experimentos import RUNS, YOLO_RUN  # noqa: E402

RESULTS = ROOT / "resultados"
FIGURES = RESULTS / "figuras"
METRICS = RESULTS / "metricas"


def save(fig, name: str):
    _note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    runs = [run for run in [*RUNS, YOLO_RUN] if load_predictions(ROOT, run, "validacao") is not None]
    if not runs:
        raise SystemExit("Nenhuma previsão encontrada; rode scripts/prever.py antes.")
    has_test = all(load_predictions(ROOT, run, "teste") is not None for run in runs)
    partitions = ["validacao", "teste"] if has_test else ["validacao"]
    print(f"Modelos: {', '.join(runs)}; partições avaliadas: {', '.join(partitions)}")

    rows, matrices, per_class, histories = [], {}, {}, {}
    for run in runs:
        meta = load_metadata(ROOT, run)
        row = {
            "experimento": run, "familia": meta.get("familia"), "variante": meta.get("variante"),
            "resolucao": meta.get("resolucao"), "parametros": meta.get("parametros"),
            "tamanho_modelo_mb": round(meta.get("tamanho_modelo_bytes", 0) / 2**20, 2),
            "lote": meta.get("lote"), "epocas_executadas": meta.get("epocas_executadas"),
            "checkpoint_epoca": meta.get("checkpoint_epoca"),
            "tempo_total_s": round(meta.get("tempo_total_segundos", np.nan), 1),
            "tempo_medio_epoca_s": round(meta.get("tempo_medio_epoca_segundos", np.nan), 1),
            "dispositivo": ", ".join(meta.get("hardware", {}).get("gpus") or [meta.get("hardware", {}).get("cpu", "")]),
        }
        for partition in partitions:
            summary, report, matrix = compute_metrics(load_predictions(ROOT, run, partition))
            suffix = "validacao" if partition == "validacao" else "teste"
            for key, value in summary.items():
                row[f"{key}_{suffix}"] = round(value, 4) if isinstance(value, float) else value
            report.to_csv(METRICS / f"{run}_{partition}_por_classe.csv", index=False, float_format="%.4f")
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(METRICS / f"{run}_{partition}_confusao.csv")
            matrices[run, partition], per_class[run, partition] = matrix, report
        rows.append(row)
        history = load_history(ROOT, run)
        if history is not None:
            histories[run] = (history, meta.get("checkpoint_epoca"))
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            plot_history(history, run, meta.get("checkpoint_epoca"), axes)
            save(fig, f"curvas_{run}.png")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "experimentos.csv", index=False)
    print(table.to_string(index=False))

    main_runs = [run for run in runs if run in RUNS]
    # Painéis com as seis execuções obrigatórias para o artigo.
    if histories:
        shown = [run for run in main_runs if run in histories]
        for metric, title in (("loss", "perda"), ("accuracy", "acurácia")):
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
            for ax, run in zip(axes.flat, shown):
                history, checkpoint = histories[run]
                epochs = history["epoch"] + 1
                ax.plot(epochs, history[metric], label="treino")
                ax.plot(epochs, history[f"val_{metric}"], label="validação")
                if checkpoint:
                    ax.axvline(checkpoint, color="crimson", ls=":", lw=1, label=f"checkpoint ({checkpoint})")
                ax.set_title(run, fontsize=10)
                ax.set_xlabel("época")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylabel(title)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
            for ax in list(axes.flat)[len(shown):]:
                ax.axis("off")
            fig.suptitle(f"Curvas de {title} por época (treino e validação)")
            save(fig, f"painel_curvas_{'perda' if metric == 'loss' else 'acuracia'}.png")

    for partition in partitions:
        for run in runs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
            plot_confusion(matrices[run, partition], axes[0], f"{run} — contagens ({partition})", normalize=False)
            plot_confusion(matrices[run, partition], axes[1], f"{run} — normalizada por classe real", normalize=True)
            save(fig, f"confusao_{run}_{partition}.png")
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), squeeze=False)
        for ax, run in zip(axes.flat, main_runs):
            plot_confusion(matrices[run, partition], ax, run, normalize=True)
        for ax in list(axes.flat)[len(main_runs):]:
            ax.axis("off")
        fig.suptitle(f"Matrizes de confusão normalizadas por classe real ({partition})")
        save(fig, f"painel_confusao_{partition}.png")

        heat = np.array([per_class[run, partition]["f1"].to_numpy() for run in runs])
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(runs) + 2))
        image = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if heat[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(CLASSES)), SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(runs)), runs, fontsize=8)
        ax.set_title(f"F1 por classe ({partition})")
        fig.colorbar(image, ax=ax, fraction=0.03)
        save(fig, f"f1_por_classe_{partition}.png")

    # Desempenho x custo: o eixo vertical usa o teste quando ele já foi avaliado.
    target = "teste" if has_test else "validacao"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for _, row in table.iterrows():
        axes[0].scatter(row["tempo_total_s"] / 60, row[f"f1_macro_{target}"], s=40)
        axes[0].annotate(row["experimento"], (row["tempo_total_s"] / 60, row[f"f1_macro_{target}"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[1].scatter(row["parametros"], row[f"f1_macro_{target}"], s=40)
        axes[1].annotate(row["experimento"], (row["parametros"], row[f"f1_macro_{target}"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("tempo total de treino (min)")
    axes[1].set_xlabel("parâmetros (escala log)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel(f"F1 macro ({target})")
        ax.grid(alpha=0.3)
    fig.suptitle("Desempenho versus custo computacional")
    save(fig, f"desempenho_custo_{target}.png")

    if has_test:
        frames = []
        for run in runs:
            frame = load_predictions(ROOT, run, "teste")
            frame.insert(0, "experimento", run)
            frames.append(frame)
        pd.concat(frames).to_csv(RESULTS / "previsoes_teste.csv", index=False, float_format="%.6f")
        # Modelo para a análise de erros: maior F1 macro de validação entre as seis execuções obrigatórias.
        best = table[table["experimento"].isin(main_runs)].sort_values(
            "f1_macro_validacao", ascending=False).iloc[0]["experimento"]
        errors = plot_errors(ROOT, load_predictions(ROOT, best, "teste"), best, FIGURES / "erros_teste.png")
        errors.insert(0, "experimento", best)
        selected = {"modelo_analise_erros": best, "criterio": "maior F1 macro de validação"}
        if YOLO_RUN in runs:
            cnn = table[table["familia"] == "CNN"].sort_values("f1_macro_validacao", ascending=False)
            selected["cnn_referencia_yolo"] = cnn.iloc[0]["experimento"]
            yolo_errors = plot_errors(ROOT, load_predictions(ROOT, YOLO_RUN, "teste"), YOLO_RUN,
                                      FIGURES / f"erros_teste_{YOLO_RUN}.png")
            yolo_errors.insert(0, "experimento", YOLO_RUN)
            errors = pd.concat([errors, yolo_errors])
        errors.drop(columns=[c for c in errors if c.startswith("p_")]).to_csv(RESULTS / "erros_teste.csv", index=False)
        (RESULTS / "selecao.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(selected, ensure_ascii=False))


if __name__ == "__main__":
    main()
