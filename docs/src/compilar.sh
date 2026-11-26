#!/usr/bin/env bash
# Gera docs/acompanhamento.pdf, docs/artigo.pdf e docs/pitch.pdf.
# Atualiza antes os números a partir de dados/splits e resultados/ (campos ainda inexistentes ficam em branco).
# Usa XeLaTeX ou Tectonic; fontes Noto Serif, Noto Sans e Hack para o artigo.
set -euo pipefail
cd "$(dirname "$0")"
python3 ../../scripts/dados_docs.py
mkdir -p build
for doc in "${@:-acompanhamento artigo pitch}"; do
  for doc_name in $doc; do
    if command -v xelatex >/dev/null 2>&1; then
      for _ in 1 2; do
        xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "$doc_name.tex" > "build/$doc_name.stdout" \
          || { tail -30 "build/$doc_name.stdout"; exit 1; }
      done
    elif command -v tectonic >/dev/null 2>&1; then
      tectonic -X compile "$doc_name.tex" --outdir build --keep-logs > "build/$doc_name.stdout" 2>&1 \
        || { tail -30 "build/$doc_name.stdout"; exit 1; }
    else
      echo 'Instale XeLaTeX ou Tectonic para compilar os documentos.' >&2
      exit 1
    fi
    cp "build/$doc_name.pdf" "../$doc_name.pdf"
    echo "docs/$doc_name.pdf"
  done
done
