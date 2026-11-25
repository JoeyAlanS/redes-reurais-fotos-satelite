#!/usr/bin/env bash
# Gera docs/acompanhamento.pdf, docs/artigo.pdf e docs/pitch.pdf.
# Atualiza antes os números a partir de dados/splits e resultados/ (campos ainda inexistentes ficam em branco).
# Requer XeLaTeX (TeX Live) e as fontes Liberation e Noto Sans.
set -euo pipefail
cd "$(dirname "$0")"
python3 ../../scripts/dados_docs.py
mkdir -p build
for doc in "${@:-acompanhamento artigo pitch}"; do
  for doc_name in $doc; do
    for _ in 1 2; do
      xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "$doc_name.tex" > "build/$doc_name.stdout" \
        || { tail -30 "build/$doc_name.stdout"; exit 1; }
    done
    cp "build/$doc_name.pdf" "../$doc_name.pdf"
    echo "docs/$doc_name.pdf ($(grep -o 'Output written.*' "build/$doc_name.stdout"))"
  done
done
