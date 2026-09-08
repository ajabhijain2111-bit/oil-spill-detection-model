import rasterio
from spill_geometry import get_spill_geometry

with rasterio.open("oil_spill_prediction.tif") as src:
    mask = src.read(1)

geometry = get_spill_geometry(mask)

print(geometry)