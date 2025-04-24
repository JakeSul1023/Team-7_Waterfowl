# csv_to_arrow.py

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import json

def convert_csv_to_arrow(csv_file, arrow_file):
    """
    Convert a CSV file to an Apache Arrow file.

    Args:
        csv_file: Path to the input CSV file.
        arrow_file: Path to the output Arrow file.
    """
    try:
        # Load the CSV
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)

        # Save to Arrow file
        with ipc.new_file(arrow_file, table.schema) as writer:
            writer.write_table(table)
        print(f"✅ Saved Arrow file to: {arrow_file}")

        print("\n🔍 Sample rows from the Arrow file:")
        print(df.head())

    except Exception as e:
        print(f"❌ Error: {e}")

def convert_csv_to_json(csv_file, json_file, orient="records", indent=4):
    """
    Convert a CSV file to a JSON file.

    Args:
        csv_file: Path to the input CSV file.
        json_file: Path to the output JSON file.
        orient: Format of JSON ('records' = list of dicts, 'columns' = column-oriented, etc.)
        indent: Indentation level for pretty-printing
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Convert to JSON
        df.to_json(json_file, orient=orient, indent=indent)
        print(f"✅ Saved JSON file to: {json_file}")

        # Optional preview
        print("\n🔍 Sample JSON Preview:")
        print(df.head().to_json(orient=orient, indent=indent))

    except Exception as e:
        print(f"❌ Error: {e}")

def convert_csv_to_geojson(csv_file, geojson_file, lat_col="forecast_lat", lon_col="forecast_lon"):
    """
    Convert a CSV file with lat/lon columns into a GeoJSON FeatureCollection.

    Args:
        csv_file: Path to the input CSV file.
        geojson_file: Path to the output GeoJSON file.
        lat_col: Name of the latitude column
        lon_col: Name of the longitude column
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df)} rows")

        features = []
        for _, row in df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row[lon_col], row[lat_col]]  # [lon, lat] per GeoJSON spec
                },
                "properties": {
                    key: row[key] for key in df.columns if key not in [lat_col, lon_col]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(geojson_file, "w") as f:
            json.dump(geojson, f, indent=4)

        print(f"✅ GeoJSON saved to {geojson_file}")
        print("🔍 Sample Feature:")
        print(json.dumps(features[0], indent=4))

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    convert_csv_to_arrow(
        csv_file="A07_A13.csv",
        arrow_file="Week_prediction.arrow"
    )
    convert_csv_to_json("A07_A13.csv", "A07_A13.json")
    convert_csv_to_geojson("A07_A13.csv", "A07_A13.geojson")
