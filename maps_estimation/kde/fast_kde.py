import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fastkde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
from time import time

Image.MAX_IMAGE_PIXELS = None


def estimate_kde(csv_path, output_base, reference_image_path):
    total_start = time()

    # Get reference image dimensions
    if reference_image_path.lower().endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(reference_image_path) as tif:
            ref_height, ref_width = tif.pages[0].shape[:2]
    else:
        with Image.open(reference_image_path) as img:
            ref_width, ref_height = img.size
    print(f"Reference: {ref_width}x{ref_height}")

    # Load points
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        df = df[df['category'] == 'Category 2']
    x, y = df['x'].values, df['y'].values
    print(f"Computing KDE for {len(x)} points...")

    # Compute FastKDE (2049 = 2^11 + 1)
    t0 = time()
    PDF = fastkde.pdf(x, y, var_names=["x", "y"], num_points=8193)
    PDF.plot()
    plt.show()


def main():
    ref_image = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X.tif"
    csv_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X_report\2025-10-16_full_detections.csv"
    output_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X_report\kde_res"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    results = estimate_kde(csv_path, output_path, ref_image)
    print(f"\nResults: {results}")


if __name__ == '__main__':
    main()
