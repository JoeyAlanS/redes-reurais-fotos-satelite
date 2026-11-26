"""Empacota o HTML e as figuras locais para importação como slides no Canva.

Execução: python scripts/empacotar_slides_canva.py
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "apresentacao_canva.html"
SAIDA = ROOT / "docs" / "apresentacao_canva.zip"
FIGURAS = [
    "eda_ambiguidades.png",
    "apresentacao_trajetoria_cnn_b_128.png",
    "apresentacao_efeito_resolucao.png",
    "apresentacao_metricas_por_classe.png",
    "apresentacao_maiores_confusoes.png",
    "erros_teste.png",
]


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    html = html.replace("../resultados/figuras/", "assets/")
    with ZipFile(SAIDA, "w", compression=ZIP_DEFLATED, compresslevel=6) as arquivo:
        arquivo.writestr("index.html", html)
        for nome in FIGURAS:
            arquivo.write(ROOT / "resultados" / "figuras" / nome, f"assets/{nome}")
    print(SAIDA)


if __name__ == "__main__":
    main()
