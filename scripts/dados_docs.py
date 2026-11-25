"""Gera docs/src/gerado/dados.tex com os números usados no artigo e nos slides.

Lê dados/splits/inventario.json, resultados/experimentos.csv e resultados/selecao.json quando existem;
o que ainda não foi gerado fica sem definição e aparece em branco nos PDFs. Só usa a biblioteca padrão.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "src" / "gerado" / "dados.tex"
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escape(text: str) -> str:
    return "".join(SPECIAL.get(char, char) for char in text)


def integer(value) -> str:
    return f"{int(float(value)):,}".replace(",", ".")


def decimal(value, places: int) -> str:
    return f"{float(value):.{places}f}".replace(".", ",")


def format_value(column: str, value: str) -> str | None:
    if value in ("", "nan", None):
        return None
    if column.startswith(("acuracia", "f1_macro", "precisao_macro", "recall_macro")):
        return decimal(value, 3)
    if column in {"parametros", "lote", "epocas_executadas", "checkpoint_epoca", "resolucao"}:
        return integer(value)
    if column == "tamanho_modelo_mb":
        return decimal(value, 1)
    if column.startswith("tempo"):
        return decimal(value, 1)
    return escape(value)


def main():
    values = {}
    inventory_path = ROOT / "dados" / "splits" / "inventario.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        values["total"] = integer(inventory["total"])
        for label, counts in inventory["contagens"].items():
            for partition, count in counts.items():
                values[f"part/{label}/{partition}"] = integer(count)
            values[f"part/{label}/total"] = integer(sum(counts.values()))
        for partition in ("treino", "validacao", "teste"):
            values[f"part/total/{partition}"] = integer(sum(c.get(partition, 0) for c in inventory["contagens"].values()))
        values["dup/exatas"] = integer(inventory["duplicatas_exatas"])
        near = inventory.get("quase_duplicatas", {})
        if near:
            values["dup/pares"] = integer(near["pares"])
            values["dup/limiar"] = integer(near["limiar"])
            values["dup/grupos"] = integer(near["grupos_com_mais_de_um_arquivo"])
            values["dup/quase"] = integer(near["pares"] - inventory["duplicatas_exatas"])
        values["problemas"] = integer(len(inventory["problemas"]))

    experiments_path = ROOT / "resultados" / "experimentos.csv"
    if experiments_path.is_file():
        with experiments_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                run = row["experimento"]
                for column, value in row.items():
                    formatted = format_value(column, value)
                    if formatted is not None and column != "experimento":
                        values[f"{run}/{column}"] = formatted
                if row.get("tempo_total_s"):
                    values[f"{run}/tempo_total_min"] = decimal(float(row["tempo_total_s"]) / 60, 1)

    selection_path = ROOT / "resultados" / "selecao.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        values["sel/modelo_erros"] = selection.get("modelo_analise_erros", "")
        if "cnn_referencia_yolo" in selection:
            values["sel/cnn_ref_yolo"] = selection["cnn_referencia_yolo"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Gerado por scripts/dados_docs.py — não editar à mão."]
    lines += [f"\\resdef{{{key}}}{{{value}}}" for key, value in sorted(values.items())]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(values)} valores em {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
