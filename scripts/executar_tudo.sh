#!/usr/bin/env bash
# Pipeline completo das seis execuções obrigatórias.
# Uso: ./scripts/executar_tudo.sh            (prepara, treina, prevê e avalia só a validação)
#      ./scripts/executar_tudo.sh --teste    (além disso, avalia o teste — rodar uma única vez, ao final)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-./scripts/python_gpu.sh}"  # no Colab: PY=python

$PY scripts/preparar_dados.py
$PY scripts/resumo_arquiteturas.py
mkdir -p resultados/historicos
for run in MLP-A-128 MLP-B-128 CNN-A-128 CNN-B-128 CNN-A-256 CNN-B-256; do
  if [[ -f "modelos/$run.json" ]]; then
    echo "$run já treinado (modelos/$run.json); apague-o para treinar de novo"
  else
    $PY scripts/treinar.py "$run" 2>&1 | tee "resultados/historicos/$run.log"
  fi
done
if [[ "${1:-}" == "--teste" ]]; then
  $PY scripts/prever.py --teste
else
  $PY scripts/prever.py
fi
$PY scripts/avaliar.py
