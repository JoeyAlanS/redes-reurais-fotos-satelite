"""Quatro arquiteturas distintas; CNNs independentes da resolução espacial."""

from tensorflow import keras


def build_model(family: str, variant: str, resolution: int) -> keras.Model:
    if family not in {"MLP", "CNN"} or variant not in {"A", "B"}:
        raise ValueError("Arquitetura deve ser MLP/CNN e variante A/B")
    if resolution not in {128, 256} or (family == "MLP" and resolution != 128):
        raise ValueError("MLP só admite 128; CNN admite 128 ou 256")
    inputs = keras.Input((resolution, resolution, 3), name="imagem_rgb")
    if family == "MLP":
        widths = (64, 32) if variant == "A" else (128, 64, 32)
        x = keras.layers.Flatten()(inputs)
        for width in widths:
            x = keras.layers.Dense(width, activation="relu")(x)
    else:
        filters = (16, 32, 64) if variant == "A" else (16, 32, 64, 96)
        x = inputs
        for count in filters:
            x = keras.layers.Conv2D(count, 3, padding="same", activation="relu")(x)
            x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="classe")(x)
    return keras.Model(inputs, outputs, name=f"{family}_{variant}_{resolution}")
