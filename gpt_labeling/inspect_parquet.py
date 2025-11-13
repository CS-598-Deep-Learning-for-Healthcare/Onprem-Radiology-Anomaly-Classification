import pyarrow.parquet as pq
import os

PARQUET_DIR = "mimic_cxr_data"

for fname in os.listdir(PARQUET_DIR):
    if fname.endswith(".parquet"):
        print("\n=== File:", fname, "===")
        table = pq.read_table(os.path.join(PARQUET_DIR, fname))
        print("Columns:", table.column_names)
        print("Schema:\n", table.schema)
        break  # inspect just one file
