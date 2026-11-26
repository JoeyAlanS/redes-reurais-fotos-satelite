# Classificação de Cenas Aéreas com Redes Neurais — Projeto 3

Projeto da Unidade III (Redes Neurais e Deep Learning) da disciplina de Ciência de Dados — Unifor.

**Autores:** Paulo Henrico Rabelo (matrícula 2312652) e Joey Alan (matrícula 2320416).

Comparamos duas MLPs e duas CNNs, treinadas do zero, no recorte de dez classes do NWPU-RESISC45, e medimos o efeito da resolução (128 × 128 versus 256 × 256) nas CNNs. As seis execuções finais e a avaliação única do teste foram concluídas. O YOLO *classify* é um bônus opcional ainda não executado.

## Dados e classes

O recorte tem 7.000 imagens RGB de 256 × 256 px, 700 por classe, cada uma com um único rótulo de cena. A ordem das classes é fixa em todo o código, métricas e matrizes:

| # | Pasta | Classe |
|---|---|---|
| 0 | `airport` | Aeroporto |
| 1 | `beach` | Praia |
| 2 | `circular_farmland` | Áreas agrícolas circulares |
| 3 | `dense_residential` | Área residencial densa |
| 4 | `desert` | Deserto |
| 5 | `forest` | Floresta |
| 6 | `harbor` | Porto |
| 7 | `mountain` | Montanha |
| 8 | `river` | Rio |
| 9 | `storage_tank` | Tanques de armazenamento |

O link OneDrive do catálogo oficial respondeu HTTP 403. Usamos uma cópia pública do mesmo ZIP no [Hugging Face](https://huggingface.co/datasets/isaaccorley/resisc45/blob/main/NWPU-RESISC45.zip), verificada por SHA-256. As imagens não vão para o Git.

```bash
mkdir -p dados/raw
curl -L --fail -o dados/raw/NWPU-RESISC45.zip \
  'https://huggingface.co/datasets/isaaccorley/resisc45/resolve/main/NWPU-RESISC45.zip'
echo 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  dados/raw/NWPU-RESISC45.zip' | sha256sum -c -
unzip -q dados/raw/NWPU-RESISC45.zip 'NWPU-RESISC45/airport/*' 'NWPU-RESISC45/beach/*' \
  'NWPU-RESISC45/circular_farmland/*' 'NWPU-RESISC45/dense_residential/*' 'NWPU-RESISC45/desert/*' \
  'NWPU-RESISC45/forest/*' 'NWPU-RESISC45/harbor/*' 'NWPU-RESISC45/mountain/*' 'NWPU-RESISC45/river/*' \
  'NWPU-RESISC45/storage_tank/*' -d dados/raw
```

## Instalação

Requer Linux, Python 3.12 e [`uv`](https://docs.astral.sh/uv/) (ou `venv` + `pip`). As dependências diretas estão em `requirements.txt` e as versões exatas usadas, em `requirements.lock.txt`.

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
cp .env.example .env
./scripts/python_gpu.sh -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

`tensorflow[and-cuda]` instala as bibliotecas CUDA no próprio ambiente; o wrapper `scripts/python_gpu.sh` as expõe ao carregador. Sem GPU, tudo roda em CPU, mas devagar: num Ryzen 5 PRO 4650U medimos ~23 s por época na MLP-A-128, ~95 s na CNN-A-128 e ~490 s na CNN-B-256, o que daria mais de 10 h para as seis execuções. Use GPU ou o Google Colab. Se faltar memória de GPU, reduza `BATCH_SIZE_*` no `.env` e registre a alteração, pois o lote faz parte do protocolo.

## Ordem de execução

| Passo | Comando | Saídas |
|---|---|---|
| 1. Inventário e partições | `./scripts/python_gpu.sh scripts/preparar_dados.py` | `dados/splits/particoes.csv`, `inventario.json`, `quase_duplicatas.csv` |
| 2. Exploração | `notebooks/01_exploracao.ipynb` | `resultados/figuras/eda_*.png` |
| 3. Arquiteturas | `./scripts/python_gpu.sh scripts/resumo_arquiteturas.py` | `resultados/arquiteturas.csv`, `resultados/arquiteturas/*.txt` |
| 4. Treinos (um por vez) | `./scripts/python_gpu.sh scripts/treinar.py CNN-A-128` | `modelos/<exp>.keras`, `modelos/<exp>.json`, `resultados/historicos/<exp>.csv` |
| 5. Previsões de validação | `./scripts/python_gpu.sh scripts/prever.py` | `resultados/previsoes/<exp>_validacao.csv` |
| 6. Avaliação | `./scripts/python_gpu.sh scripts/avaliar.py` | `resultados/experimentos.csv`, `metricas/`, `figuras/` |
| 7. Teste (**uma única vez, ao final**) | `./scripts/python_gpu.sh scripts/prever.py --teste` e de novo `scripts/avaliar.py` | `previsoes_teste.csv`, `erros_teste.csv`, `selecao.json`, figuras de teste |
| 8. Resultados | `notebooks/02_resultados.ipynb` | tabelas e figuras consolidadas |

`./scripts/executar_tudo.sh` executa os passos 1, 3, 4, 5 e 6 em sequência e pula modelos já treinados. Com `--teste`, inclui o passo 7. No Google Colab, use `notebooks/00_colab_treino.ipynb`, que baixa os dados, instala as dependências, roda os seis treinos em GPU e copia os resultados para o Drive.

**Para avaliar sem treinar:** as previsões, métricas, históricos e figuras gerados estão em `resultados/`. Os checkpoints `.keras` estão disponíveis localmente em `modelos/`; para reproduzir as previsões em outra máquina, é necessário obtê-los junto à equipe ou executar os treinos.

## Preparação e prevenção de vazamento

- `preparar_dados.py` abre todas as imagens, confere dimensões (256 × 256) e modo de cor (RGB) e calcula SHA-256. Os originais nunca são alterados.
- Duplicatas exatas (SHA-256) e quase-duplicatas formam grupos. As quase-duplicatas são detectadas por dHash de 256 bits comparado nas 8 orientações, então rotações e espelhamentos também contam; o limiar de Hamming é definido por `NEAR_DUP_THRESHOLD`. Cada grupo é sorteado como uma unidade, de modo que todas as versões de uma cena ficam na mesma partição.
- A divisão é 70/15/15, estratificada por classe, com semente 42 e caminhos ordenados antes do sorteio. Sem grupos, seriam 490/105/105 imagens por classe; as contagens finais estão em `dados/splits/inventario.json`.
- A partição de cada arquivo fica salva em `particoes.csv`, com caminhos relativos, e é reutilizada por todos os experimentos e pelo bônus.
- A exploração visual usa só o treino. Validação e teste aparecem apenas em contagens e checagens de integridade e de sobreposição.
- A normalização é RGB redimensionado a partir do original (bilinear com antialias) e pixels ÷ 255. Nenhuma estatística é aprendida. O aumento de dados (espelhamentos e giros de 90°) atua apenas no treino.

## Arquiteturas e hipóteses

As quatro arquiteturas estão em `src/projeto3/models.py`. As camadas e os parâmetros de cada uma ficam em `resultados/arquiteturas/`.

| Modelo | Estrutura | Hipótese |
|---|---|---|
| MLP A | vetor 49.152 → Dense 256 → Dense 10 | Uma camada oculta larga; referência sem viés espacial. |
| MLP B | vetor 49.152 → Dense 256 → 128 → 64 → Dense 10 | Quase o mesmo número de parâmetros de A, com três camadas ocultas. Isola o efeito da **profundidade** numa MLP. |
| CNN A | 3 blocos [Conv 3×3 → BN → ReLU → MaxPool] (32/64/128) → GAP → Dense 10 | Convolução com pesos compartilhados capta textura local com uma fração dos parâmetros da MLP. |
| CNN B | 4 blocos (32/64/128/256) → GAP → Dense 128 → Dense 10 | Um bloco a mais dobra o campo receptivo e adiciona capacidade. Espera-se ganho maior em 256 px, onde o campo receptivo da CNN A cobre uma fração menor da cena. |

As camadas ocultas das MLPs usam BatchNorm e Dropout de 0,3; as CNNs têm Dropout de 0,3 antes da saída. O BatchNorm usa `momentum=0.9` em vez do padrão 0,99. Num teste com o padrão, depois de 2 épocas (cerca de 300 lotes) a variância móvel da primeira camada ainda era 0,048, quase toda herdada do valor inicial 1, enquanto a variância real dos lotes era 0,003. A validação colapsava numa única classe (11% de acurácia, contra 66% no treino). Com 0,9, as estatísticas de inferência acompanham as do treino já na primeira época. As CNNs usam `GlobalAveragePooling2D`, então filtros, kernels, blocos e cabeça, e portanto os parâmetros, são idênticos em 128 e 256 px; `resumo_arquiteturas.py` verifica isso. Nenhum peso é reaproveitado entre resoluções.

## Protocolo de treinamento

| Item | Valor |
|---|---|
| Biblioteca | TensorFlow/Keras (versões em `requirements.lock.txt` e em cada `modelos/<exp>.json`) |
| Inicialização | pesos novos em todas as execuções, sem pré-treinamento |
| Perda / otimizador | entropia cruzada categórica esparsa / Adam, taxa 0,001 |
| Lote executado | 8 nas duas resoluções (`BATCH_SIZE_128`, `BATCH_SIZE_256`), para caber na RTX 2050 de 4 GB |
| Épocas executadas | limite de 12 (`MAX_EPOCHS`) |
| Seleção e parada | F1 macro de validação calculado a cada época; checkpoint da época de maior F1; parada após 4 épocas sem melhora (`PATIENCE`) |
| Semente | 42 (`keras.utils.set_random_seed`, embaralhamento e sorteio das partições) |
| Aumento | espelhamento horizontal/vertical e rotação de 90° aleatórios, só no treino (`AUGMENTATION=1`) |
| Custo | `time.perf_counter` em volta de `model.fit`, mais o tempo de cada época; exclui download e preparação |

Pré-processamento, semente, otimizador, taxa, lote e política de parada são os mesmos para cada CNN nas duas resoluções, o que isola o efeito da resolução. O teste não é lido durante o treino nem na escolha de configurações. Ele é previsto uma única vez, com `prever.py --teste`, depois de encerradas as escolhas. Buscas exploratórias extras devem ficar separadas das seis execuções finais; por exemplo, rode-as com outro `.env` e não sobrescreva `modelos/`.

Cada `modelos/<exp>.json` registra a arquitetura, os hiperparâmetros, o critério de parada, a época do checkpoint, o F1 macro de validação do checkpoint recuperado, os tempos (total e médio por época), o tamanho do arquivo, o hardware (GPU/CPU) e as versões das bibliotecas.

## Resultados dos seis treinos

| Modelo | F1 macro validação | F1 macro teste | Tempo total (s) |
|---|---:|---:|---:|
| MLP-A-128 | 0,5409 | 0,5295 | 96,5 |
| MLP-B-128 | 0,5325 | 0,5261 | 118,2 |
| CNN-A-128 | 0,8169 | 0,8041 | 157,1 |
| CNN-B-128 | **0,8195** | **0,8410** | 182,3 |
| CNN-A-256 | 0,8067 | 0,8095 | 380,3 |
| CNN-B-256 | 0,8170 | 0,8204 | 434,6 |

`CNN-B-128` foi selecionada pelo maior F1 macro de validação. O teste foi previsto uma vez após essa escolha; o lote de **inferência** foi 4 por causa da memória da GPU e não altera o protocolo de treino. O artigo em `docs/artigo.pdf` discute custo, curvas, confusões e erros. Os números completos estão em `resultados/experimentos.csv` e `resultados/metricas/`.

## Evidências geradas (`scripts/avaliar.py`)

- `resultados/experimentos.csv`: por modelo, acurácia, F1 macro, precisão e recall macro (validação e teste), parâmetros, tamanho, épocas, checkpoint, tempos, lote e dispositivo.
- `resultados/metricas/<exp>_<particao>_por_classe.csv`: precisão, recall, F1 e suporte nas dez classes. Quando uma classe nunca é prevista, a precisão é 0/0; usamos `zero_division=0` e listamos essas classes em `classes_sem_previsao_*`.
- `resultados/metricas/<exp>_<particao>_confusao.csv` e as figuras `confusao_*.png`: contagens e normalização por classe real, com reais nas linhas e previstas nas colunas.
- `figuras/curvas_*.png` e `painel_curvas_*.png`: perda e acurácia de treino e validação por época, com o checkpoint marcado.
- `figuras/desempenho_custo_*.png` e `f1_por_classe_*.png`.
- `erros_teste.csv` e `figuras/erros_teste.png`: até seis erros de teste, os de maior confiança, do modelo com maior F1 macro de validação (`selecao.json`).
- `resultados/previsoes_teste.csv`: previsões de teste de todos os modelos, com confiança e as dez probabilidades.

## Bônus: YOLO classify

Usa um ambiente separado, para não misturar CUDA do PyTorch com a do TensorFlow:

```bash
uv venv --python 3.12 .venv-yolo
uv pip install --python .venv-yolo/bin/python -r requirements-yolo.txt
.venv-yolo/bin/python scripts/yolo_bonus.py --pesos yolo11n-cls.pt --epocas 30 --imgsz 256 --lote 32
./scripts/python_gpu.sh scripts/avaliar.py
```

O script monta `dados/yolo/{train,val,test}/<classe>/` com links simbólicos para as mesmas partições, ajusta os pesos pré-treinados no ImageNet e salva `modelos/YOLO-cls.pt`, o histórico e as previsões no mesmo formato. Os argumentos completos do treino ficam em `modelos/YOLO-cls.json`. A Ultralytics escolhe o checkpoint pela acurácia top-1 de validação, e não pelo F1 macro; registramos o F1 macro desse checkpoint. `avaliar.py` inclui o YOLO nas tabelas e compara com a CNN de maior F1 macro de validação (`selecao.json`).

## Modelos salvos

Os seis arquivos `.keras` são ignorados pelo Git por causa do tamanho (as MLPs têm cerca de 49 MB cada). Eles estão anexados à [release `resultados-v1`](https://github.com/JoeyAlanS/redes-reurais-fotos-satelite/releases/tag/resultados-v1) deste repositório privado; é necessário acesso ao repositório para baixá-los. Para reproduzir as previsões sem treinar:

```bash
gh release download resultados-v1 -p '*.keras' -D modelos
cd modelos && sha256sum -c SHA256SUMS.txt && cd ..
```

Os metadados, resumos e hashes SHA-256 ficam versionados em `modelos/`. O professor precisará ser adicionado como colaborador ao repositório privado para acessar a release.

## Documentos (`docs/`)

`docs/acompanhamento.pdf`, `docs/artigo.pdf` e `docs/pitch.pdf` são gerados a partir dos fontes LaTeX em `docs/src/`:

```bash
./docs/src/compilar.sh            # os três; ou: ./docs/src/compilar.sh artigo
```

O script roda `scripts/dados_docs.py`, que lê `dados/splits/inventario.json`, `resultados/experimentos.csv` e `resultados/selecao.json` e grava os números em `docs/src/gerado/dados.tex`. Depois, compila com XeLaTeX ou Tectonic. O artigo foi atualizado com os seis resultados, gráficos e discussão; acompanhamento e pitch ainda exigem revisão dos autores antes da entrega. O acompanhamento não deve mostrar métricas de teste.

## Estrutura

```text
dados/raw/            ZIP e imagens (fora do Git)
dados/splits/         particoes.csv, inventario.json, quase_duplicatas.csv
notebooks/            00_colab_treino.ipynb, 01_exploracao.ipynb, 02_resultados.ipynb
src/projeto3/         classes, preparação, dataset, modelos, avaliação
scripts/              preparar_dados, resumo_arquiteturas, treinar, prever, avaliar, yolo_bonus, dados_docs, executar_tudo
modelos/              checkpoints (fora do Git) e metadados .json
resultados/           experimentos.csv, previsoes_teste.csv, historicos/, previsoes/, metricas/, figuras/
docs/                 acompanhamento.pdf, artigo.pdf, pitch.pdf; fontes LaTeX em docs/src/
```

## Referências

- CHENG, G.; HAN, J.; LU, X. Remote sensing image scene classification: benchmark and state of the art. *Proceedings of the IEEE*, v. 105, n. 10, p. 1865–1883, 2017.
- TensorFlow: [classificação de imagens](https://www.tensorflow.org/tutorials/images/classification) e [catálogo RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45).
- Keras: [GlobalAveragePooling2D](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/) e [callbacks](https://keras.io/api/callbacks/).
- scikit-learn: [métricas de classificação](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
- Ultralytics: [YOLO classify](https://docs.ultralytics.com/tasks/classify/).
- KRAWETZ, N. *Kind of like that* (dHash), 2013. Disponível em: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html.
