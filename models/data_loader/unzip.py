import os
import zipfile

ZIP_DIR = "/home/ssl30/Desktop/geospace/datasets/check"   # folder where your zip files are
EXTRACT_DIR = "/home/ssl30/Desktop/geospace/data"

import subprocess

os.makedirs(EXTRACT_DIR, exist_ok=True)

def unzip_all():
    for file in os.listdir(ZIP_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(ZIP_DIR, file)
            extract_path = os.path.join(EXTRACT_DIR, file.replace(".zip", ""))

            os.makedirs(extract_path, exist_ok=True)

            print(f"Extracting: {file}")

            subprocess.run(["unzip", "-o", zip_path, "-d", extract_path])

    print("\n✅ Done extracting!")

if __name__ == "__main__":
    unzip_all()