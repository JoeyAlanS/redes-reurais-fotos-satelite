"""Ordem fixa definida na seção 2 do enunciado."""

CLASSES = (
    "airport", "beach", "circular_farmland", "dense_residential", "desert",
    "forest", "harbor", "mountain", "river", "storage_tank",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
NOMES_PT = {
    "airport": "Aeroporto", "beach": "Praia", "circular_farmland": "Agrícola circular",
    "dense_residential": "Residencial densa", "desert": "Deserto", "forest": "Floresta",
    "harbor": "Porto", "mountain": "Montanha", "river": "Rio", "storage_tank": "Tanques",
}
