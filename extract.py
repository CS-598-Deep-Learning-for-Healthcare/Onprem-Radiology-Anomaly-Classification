import zipfile
import os
from pathlib import Path

ZIP_PATH = "G:\Jupyterstuff\CS598DLH\Onprem-Radiology-Anomaly-Classification\mimic-cxr-reports.zip"
OUTPUT_DIR = Path("mimic_cxr_reports")

def extract_mimic_cxr_reports(zip_path, output_dir):
    """
    Extract only .txt radiology reports from mimic-cxr-reports.zip.
    Handles PhysioNet's weird partial/directory-embedded structure.
    """

    output_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()

        print(f"Total items in ZIP: {len(members)}")

        txt_files = [m for m in members if m.lower().endswith(".txt")]

        print(f"Found {len(txt_files)} .txt report files.")

        for member in txt_files:
            out_path = output_dir / member

            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

        print(f"Extraction complete. Reports saved under: {output_dir}")

# Run it
extract_mimic_cxr_reports(ZIP_PATH, OUTPUT_DIR)