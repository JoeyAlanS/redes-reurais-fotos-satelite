# redes-reurais-fotos-satelite

Base do projeto da Unidade III de Ciência de Dados. O objetivo do enunciado é comparar duas MLPs e duas CNNs no recorte de dez classes do NWPU-RESISC45, com seis treinos independentes. **O projeto usa arquivos de imagem; o PDF não especifica banco SQL, tabelas, API ou autenticação.** Por isso não há credenciais, migrations ou servidor nesta base.

## Requisitos

- Linux com Python 3.12 e `uv` (ou `venv`/`pip`). Dependências diretas estão em `requirements.txt`; as versões instaladas nesta etapa, inclusive CUDA, estão em `requirements.lock.txt`.
- Espaço para o ZIP de 427.389.445 bytes e as 7.000 imagens extraídas.
- GPU NVIDIA é opcional. Nesta máquina, a RTX 2050 tem 4 GB; lotes padrão de 16 (128 px) e 8 (256 px) limitam o consumo. Se faltar memória, reduza os lotes no `.env` e registre a alteração na comparação.

## Instalação

```bash
cd redes-reurais-fotos-satelite
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
cp .env.example .env
./scripts/python_gpu.sh -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Em Linux, `tensorflow[and-cuda]` instala as bibliotecas CUDA de usuário. O wrapper `python_gpu.sh` expõe essas bibliotecas ao carregador e deixa o `ptxas` no `PATH`; foi necessário nesta máquina para a GPU aparecer. Verifique `nvidia-smi` se a lista continuar vazia. A execução em CPU também é possível, mas mais lenta.

## Dados e preparação

O link OneDrive indicado no pedido respondeu **HTTP 403** neste ambiente. Foi obtida uma cópia pública do mesmo conjunto em [Hugging Face](https://huggingface.co/datasets/isaaccorley/resisc45/blob/main/NWPU-RESISC45.zip). O [catálogo TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/resisc45) também descreve a distribuição original via OneDrive. O arquivo recebido é um ZIP, não um dump de banco.

```bash
mkdir -p dados/raw
curl -L --fail -o dados/raw/NWPU-RESISC45.zip \
  'https://huggingface.co/datasets/isaaccorley/resisc45/resolve/main/NWPU-RESISC45.zip'
echo 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  dados/raw/NWPU-RESISC45.zip' | sha256sum -c -
unzip -tqq dados/raw/NWPU-RESISC45.zip
unzip -q dados/raw/NWPU-RESISC45.zip 'NWPU-RESISC45/airport/*' 'NWPU-RESISC45/beach/*' 'NWPU-RESISC45/circular_farmland/*' 'NWPU-RESISC45/dense_residential/*' 'NWPU-RESISC45/desert/*' 'NWPU-RESISC45/forest/*' 'NWPU-RESISC45/harbor/*' 'NWPU-RESISC45/mountain/*' 'NWPU-RESISC45/river/*' 'NWPU-RESISC45/storage_tank/*' -d dados/raw
.venv/bin/python scripts/preparar_dados.py
```

O script lê as dez pastas, testa a abertura das imagens, registra dimensões e canais, calcula hashes SHA-256 e salva `dados/splits/inventario.json` e `dados/splits/particoes.csv`. Os caminhos são relativos à raiz. A divisão usa semente 42, ordem de arquivos fixa e grupos de duplicatas **exatas**. Sem exclusões ou duplicatas, cada classe deve ter 490 imagens de treino, 105 de validação e 105 de teste. Examine o relatório antes de treinar; versões visualmente semelhantes exigem revisão manual para evitar vazamento entre partições. Os originais não são alterados.

Ordem fixa das classes: `airport`, `beach`, `circular_farmland`, `dense_residential`, `desert`, `forest`, `harbor`, `mountain`, `river`, `storage_tank`.

## Treinamento inicial

```bash
./scripts/python_gpu.sh scripts/treinar.py MLP-A-128
./scripts/python_gpu.sh scripts/treinar.py MLP-B-128
./scripts/python_gpu.sh scripts/treinar.py CNN-A-128
./scripts/python_gpu.sh scripts/treinar.py CNN-B-128
./scripts/python_gpu.sh scripts/treinar.py CNN-A-256
./scripts/python_gpu.sh scripts/treinar.py CNN-B-256
```

Os modelos recebem RGB redimensionado diretamente do original e pixels divididos por 255. Não há aumento de dados. A semente é 42, o otimizador é Adam com taxa 0,001 e a perda é entropia cruzada para rótulos inteiros. Cada invocação constrói pesos novos. A parada usa perda de validação com paciência 3; o script salva o melhor checkpoint, histórico CSV e metadados JSON, incluindo F1 macro do checkpoint recuperado. O teste não é lido durante o treino. As CNNs usam `GlobalAveragePooling2D`; A tem três blocos convolucionais (16/32/64 filtros) e B tem quatro (16/32/64/96). Os parâmetros de cada CNN permanecem iguais em 128 e 256 px. As MLPs usam 64/32 e 128/64/32 neurônios, respectivamente, depois do achatamento. Essas são escolhas iniciais leves para 4 GB e devem ser justificadas pelos integrantes.

## Estrutura

```text
dados/raw/          ZIP e imagens locais; ignorados pelo Git
dados/splits/       Inventário e divisão persistida
src/projeto3/       Classes, preparação, carregamento e arquiteturas
scripts/            Comandos de preparação e treino
notebooks/          Exploração e análise futuras
modelos/            Checkpoints .keras e metadados
resultados/         Históricos, previsões, métricas e figuras futuras
docs/               PDFs de acompanhamento, artigo e pitch futuros
```

Esta etapa prepara o ambiente e a execução dos experimentos; ainda não executa os seis treinos nem produz métricas de teste, figuras, artigo ou pitch. Conforme o PDF, a seleção será feita na validação. Somente depois de concluir as escolhas, os seis modelos salvos deverão ser avaliados no mesmo teste, uma única vez, com as métricas e evidências exigidas. Não use os resultados de teste para reajustar modelos.

## Referências

- Enunciado fornecido: *Projeto 3 — Classificação de Cenas Aéreas com Redes Neurais*, pp. 3–8.
- [TensorFlow: instalação](https://www.tensorflow.org/install/pip).
- [Keras: callbacks e checkpoint](https://keras.io/api/callbacks/).
