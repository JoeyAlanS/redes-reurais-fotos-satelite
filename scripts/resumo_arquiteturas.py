"""Resumo e contagem de parâmetros das quatro arquiteturas, sem treinar.

Confirma que cada CNN tem os mesmos parâmetros em 128 e 256 px.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from projeto3.experimentos import RUNS  # noqa: E402
from projeto3.models import build_model, describe  # noqa: E402


def main():
    output = ROOT / "resultados" / "arquiteturas"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, (family, variant, resolution) in RUNS.items():
        model = build_model(family, variant, resolution)
        (output / f"{run}.txt").write_text(describe(model), encoding="utf-8")
        layers = [layer for layer in model.layers if layer.count_params() or layer.__class__.__name__ != "InputLayer"]
        rows.append({
            "experimento": run, "familia": family, "variante": variant, "resolucao": resolution,
            "camadas": len(layers), "parametros": model.count_params(),
            "parametros_treinaveis": sum(int(w.numpy().size) for w in model.trainable_weights),
        })
    table = pd.DataFrame(rows)
    for family_variant, group in table[table["familia"] == "CNN"].groupby("variante"):
        if group["parametros"].nunique() != 1:
            raise AssertionError(f"CNN {family_variant} muda de parâmetros entre resoluções")
    table.to_csv(ROOT / "resultados" / "arquiteturas.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
