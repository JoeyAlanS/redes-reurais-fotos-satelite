"""Ordem fixa definida na seção 2 do enunciado."""

CLASSES = (
    "airport", "beach", "circular_farmland", "dense_residential", "desert",
    "forest", "harbor", "mountain", "river", "storage_tank",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
