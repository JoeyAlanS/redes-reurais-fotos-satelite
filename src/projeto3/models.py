"""Quatro arquiteturas distintas; CNNs independentes da resolução espacial.

MLP A e B têm quase o mesmo número de parâmetros (dominado pela primeira camada densa),
então a comparação isola a profundidade. CNN B acrescenta um quarto bloco convolucional,
que dobra o campo receptivo, e uma camada densa na cabeça.
"""

from tensorflow import keras

MLP_UNITS = {"A": (256,), "B": (256, 128, 64)}
CNN_FILTERS = {"A": (32, 64, 128), "B": (32, 64, 128, 256)}
CNN_HEAD = {"A": (), "B": (128,)}
DROPOUT = 0.3
# O padrão do Keras (0,99) leva milhares de passos para esquecer a variância inicial 1; com ~150 lotes
# por época as estatísticas de inferência ficavam defasadas e a validação colapsava nas primeiras épocas.
BN_MOMENTUM = 0.9


def build_model(family: str, variant: str, resolution: int, num_classes: int = 10) -> keras.Model:
    if family not in {"MLP", "CNN"} or variant not in {"A", "B"}:
        raise ValueError("Arquitetura deve ser MLP/CNN e variante A/B")
    if resolution not in {128, 256} or (family == "MLP" and resolution != 128):
        raise ValueError("MLP só admite 128; CNN admite 128 ou 256")
    inputs = keras.Input((resolution, resolution, 3), name="imagem_rgb")
    if family == "MLP":
        x = keras.layers.Flatten(name="vetor")(inputs)
        for index, units in enumerate(MLP_UNITS[variant], start=1):
            x = keras.layers.Dense(units, use_bias=False, name=f"densa{index}")(x)
            x = keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn{index}")(x)
            x = keras.layers.ReLU(name=f"relu{index}")(x)
            x = keras.layers.Dropout(DROPOUT, name=f"dropout{index}")(x)
    else:
        x = inputs
        for index, filters in enumerate(CNN_FILTERS[variant], start=1):
            x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
            x = keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn{index}")(x)
            x = keras.layers.ReLU(name=f"relu{index}")(x)
            x = keras.layers.MaxPooling2D(name=f"pool{index}")(x)
        # Agregação global: a cabeça não depende de 128 ou 256 px, logo os parâmetros são iguais.
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        for index, units in enumerate(CNN_HEAD[variant], start=1):
            x = keras.layers.Dense(units, activation="relu", name=f"densa{index}")(x)
        x = keras.layers.Dropout(DROPOUT, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classe")(x)
    return keras.Model(inputs, outputs, name=f"{family}_{variant}_{resolution}")


def describe(model: keras.Model) -> str:
    """Resumo em texto simples (camada, tipo, saída, parâmetros), sem depender da largura do terminal."""
    def number(value):
        return f"{value:,}".replace(",", ".")

    lines = [f"Modelo: {model.name}", f"{'camada':<16}{'tipo':<26}{'saída':<20}{'parâmetros':>14}", "-" * 76]
    for layer in model.layers:
        shape = " x ".join(str(d) for d in layer.output.shape[1:])
        lines.append(f"{layer.name:<16}{layer.__class__.__name__:<26}{shape:<20}{number(layer.count_params()):>14}")
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    lines += ["-" * 76, f"Total de parâmetros: {number(model.count_params())}",
              f"Treináveis: {number(trainable)} | não treináveis (médias/variâncias do BatchNorm): "
              f"{number(model.count_params() - trainable)}"]
    return "\n".join(lines) + "\n"
