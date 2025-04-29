import json
from shapely.geometry import shape, mapping
from shapely import ops
from pyproj import Transformer

# Set up the transformer: from EPSG:3857 (meters) to EPSG:4326 (degrees)
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Load your layer.json file
with open("layer.json", "r") as f:
    geojson_data = json.load(f)

# Reproject all features in the GeoJSON
for feature in geojson_data["features"]:
    geometry = shape(feature["geometry"])  # convert to shapely geometry
    transformed = ops.transform(transformer.transform, geometry)  # reproject geometry
    feature["geometry"] = mapping(transformed)  # convert back to GeoJSON format

# Save the reprojected data
with open("layer_reprojected.geojson", "w") as f:
    json.dump(geojson_data, f, indent=2)

print("Reprojection complete. Output written to layer_reprojected.geojson.")
