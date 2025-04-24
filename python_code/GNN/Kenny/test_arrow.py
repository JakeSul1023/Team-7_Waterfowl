# test_arrow_export.py

import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

def export_predictions_to_arrow(prediction_results, output_file="SpringDucks_Predictions.arrow"):
    """
    Export prediction results to an Apache Arrow file.
    
    Args:
        prediction_results: List of dictionaries containing prediction data.
        output_file: File path for the output .arrow file.
    """
    if not prediction_results:
        print("❌ No prediction results to export.")
        return

    df = pd.DataFrame(prediction_results)
    arrow_table = pa.Table.from_pandas(df)

    with ipc.new_file(output_file, arrow_table.schema) as writer:
        writer.write_table(arrow_table)

    print(f"✅ Apache Arrow file saved to {output_file}")


def read_arrow_file(file_path):
    """
    Read and return the contents of an Apache Arrow file as a DataFrame.
    """
    with open(file_path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()
    return table.to_pandas()


# === Sample practice data ===
sample_predictions = [
    {
        "duck_id": "001",
        "base_timestamp": "2024-09-09 04:19:05",
        "forecast_timestamp": "2024-09-10 04:19:05",
        "start_lat": 34.05,
        "start_lon": -118.25,
        "forecast_lat": 34.15,
        "forecast_lon": -118.35
    },
    {
        "duck_id": "002",
        "base_timestamp": "2024-09-09 07:00:00",
        "forecast_timestamp": "2024-09-10 07:00:00",
        "start_lat": 40.71,
        "start_lon": -74.00,
        "forecast_lat": 40.73,
        "forecast_lon": -74.02
    }
]

# File name
arrow_file = "test_duck_predictions.arrow"

# Export and display
export_predictions_to_arrow(sample_predictions, output_file=arrow_file)

print("\n🔍 Contents of the Arrow file:")
df = read_arrow_file(arrow_file)
print(df)
