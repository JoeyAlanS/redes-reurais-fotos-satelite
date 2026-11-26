"""Gera quatro gráficos de apresentação a partir dos resultados medidos.

Execução: python scripts/graficos_apresentacao.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "resultados"
SAIDA = RESULTADOS / "figuras"
AZUL = "#4e82b5"
CIANO = "#4b9eb7"
ESCURO = "#172735"
LARANJA = "#e47b59"
CINZA = "#566675"
FUNDO = "#ffffff"
CLASSES = {
    "airport": "Aeroporto",
    "beach": "Praia",
    "circular_farmland": "Agrícola circular",
    "dense_residential": "Residencial denso",
    "desert": "Deserto",
    "forest": "Floresta",
    "harbor": "Porto",
    "mountain": "Montanha",
    "river": "Rio",
    "storage_tank": "Tanques",
}


def ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": ESCURO,
        "text.color": ESCURO,
        "axes.edgecolor": "#c7d9e5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "savefig.facecolor": FUNDO,
    })


def salvar(nome: str) -> None:
    plt.savefig(SAIDA / nome, dpi=190, bbox_inches="tight", pad_inches=0.22)
    plt.close()


def rodape(fig: plt.Figure, fonte: str) -> None:
    fig.text(0.01, 0.01, fonte, color=CINZA, fontsize=8)


def efeito_resolucao(experimentos: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    for variante, cor in (("A", AZUL), ("B", CIANO)):
        linhas = [experimentos[f"CNN-{variante}-{r}"] for r in (128, 256)]
        f1 = [float(linha["f1_macro_teste"]) for linha in linhas]
        tempo = [float(linha["tempo_total_s"]) for linha in linhas]
        ax.plot([128, 256], f1, marker="o", markersize=12, linewidth=3, color=cor,
                label=f"CNN {variante}")
        for x, y in zip((128, 256), f1):
            ax.annotate(f"{y:.1%}".replace(".", ","), (x, y), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontweight="bold", color=cor)
        delta = (f1[1] - f1[0]) * 100
        ax.text(192, 0.862 if variante == "B" else 0.788,
                f"{delta:+.1f} p.p. de F1  •  treino {tempo[1] / tempo[0]:.1f}× mais longo".replace(".", ","),
                color=cor, ha="center", fontsize=10)
    ax.set_xticks([128, 256], ["128 × 128 px", "256 × 256 px"])
    ax.set_xlim(100, 284)
    ax.set_ylim(0.77, 0.89)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e4eef4")
    ax.set_ylabel("F1 macro no teste")
    ax.set_title("Mais resolução não garantiu melhor resultado", loc="left", fontweight="bold", fontsize=18, pad=20)
    ax.legend(frameon=False, loc="lower left")
    rodape(fig, "Fonte: resultados/experimentos.csv • 10 classes, mesmo conjunto de teste")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar("apresentacao_efeito_resolucao.png")


def metricas_por_classe() -> None:
    linhas = ler_csv(RESULTADOS / "metricas" / "CNN-B-128_teste_por_classe.csv")
    linhas.sort(key=lambda linha: float(linha["recall"]))
    nomes = [CLASSES[linha["classe"]] for linha in linhas]
    recalls = [float(linha["recall"]) for linha in linhas]
    precisao = [float(linha["precisao"]) for linha in linhas]
    y = list(range(len(linhas)))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.barh([i + 0.18 for i in y], recalls, height=0.34, color=CIANO, label="Recall")
    ax.barh([i - 0.18 for i in y], precisao, height=0.34, color=AZUL, label="Precisão")
    ax.set_yticks(y, nomes)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color="#e4eef4")
    ax.set_axisbelow(True)
    ax.set_title("Desempenho por classe — CNN B, 128 px", loc="left", fontweight="bold", fontsize=18, pad=20)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    rodape(fig, "Fonte: resultados/metricas/CNN-B-128_teste_por_classe.csv • 105 imagens por classe")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar("apresentacao_metricas_por_classe.png")


def maiores_confusoes() -> None:
    caminho = RESULTADOS / "metricas" / "CNN-B-128_teste_confusao.csv"
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo)
        colunas = next(leitor)[1:]
        linhas = [(linha[0], [int(x) for x in linha[1:]]) for linha in leitor]
    erros = [(valor, CLASSES[real], CLASSES[predita])
             for real, valores in linhas for predita, valor in zip(colunas, valores)
             if real != predita and valor > 0]
    erros.sort(reverse=True)
    principais = list(reversed(erros[:8]))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    barras = ax.barh(range(len(principais)), [x[0] for x in principais], color=CIANO)
    ax.set_yticks(range(len(principais)), [f"{real} → {predita}" for _, real, predita in principais])
    ax.set_xlim(0, max(x[0] for x in principais) * 1.18)
    ax.bar_label(barras, padding=5, color=ESCURO, fontweight="bold")
    ax.set_xlabel("Imagens classificadas incorretamente")
    ax.set_title("Confusões mais frequentes — CNN B, 128 px", loc="left", fontweight="bold", fontsize=18, pad=20)
    ax.grid(axis="x", color="#e4eef4")
    ax.set_axisbelow(True)
    rodape(fig, "Fonte: resultados/metricas/CNN-B-128_teste_confusao.csv • linhas = classe real, colunas = predição")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar("apresentacao_maiores_confusoes.png")


def trajetoria() -> None:
    linhas = ler_csv(RESULTADOS / "historicos" / "CNN-B-128.csv")
    epocas = [int(linha["epoch"]) + 1 for linha in linhas]
    serie = [
        ("F1 macro de validação", "val_f1_macro", CIANO, "o"),
        ("Acurácia de validação", "val_accuracy", AZUL, "s"),
        ("Acurácia de treino", "accuracy", CINZA, "^"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    for nome, coluna, cor, marcador in serie:
        ax.plot(epocas, [float(linha[coluna]) for linha in linhas], color=cor,
                linewidth=2.5, marker=marcador, markersize=6, label=nome)
    melhor = max(linhas, key=lambda linha: float(linha["val_f1_macro"]))
    melhor_epoca = int(melhor["epoch"]) + 1
    ax.axvline(melhor_epoca, color=LARANJA, linestyle="--", linewidth=1.6)
    ax.annotate(f"checkpoint: época {melhor_epoca}", (melhor_epoca, float(melhor["val_f1_macro"])),
                xytext=(-88, 20), textcoords="offset points", color=LARANJA, fontweight="bold")
    ax.set_xticks(epocas)
    ax.set_ylim(0.35, 0.95)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Época")
    ax.set_ylabel("Métrica")
    ax.set_title("Evolução do modelo selecionado", loc="left", fontweight="bold", fontsize=18, pad=20)
    ax.grid(axis="y", color="#e4eef4")
    ax.legend(frameon=False, loc="lower right")
    rodape(fig, "Fonte: resultados/historicos/CNN-B-128.csv • checkpoint escolhido pelo maior F1 macro de validação")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar("apresentacao_trajetoria_cnn_b_128.png")


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    estilo()
    experimentos = {linha["experimento"]: linha for linha in ler_csv(RESULTADOS / "experimentos.csv")}
    efeito_resolucao(experimentos)
    metricas_por_classe()
    maiores_confusoes()
    trajetoria()
    for caminho in sorted(SAIDA.glob("apresentacao_*.png")):
        print(caminho.relative_to(ROOT))


if __name__ == "__main__":
    main()
