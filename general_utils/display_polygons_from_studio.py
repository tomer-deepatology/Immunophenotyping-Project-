import pandas as pd
import tifffile
import numpy as np
import cv2
from tqdm import tqdm
from osgeo import gdal
import tempfile
import os
import json

# ===== PARAMETERS =====
csv_path = r"C:\Users\tomer\Desktop\data\demo_sunday\report\2025-10-30_full_detections.csv"
input_image_path = r"C:\Users\tomer\Desktop\data\demo_sunday\demo_best_resolution.tif"
output_image_path = r"C:\Users\tomer\Desktop\data\demo_sunday\detections_overlay.tif"
color = (255, 0, 0)  # Red in RGB
thickness = 2  # Line thickness (-1 for filled)
jpeg_quality = 90
overview_levels = [2, 4, 8, 16, 32]
# ======================

# Load CSV
print("Loading detections...")
df = pd.read_csv(csv_path)
print(f"Total detections: {len(df)}")

# Filter by category
df = df[df['category'] == 'Category 2']
print(f"Detections after filtering: {len(df)}")

# Load image
print("Loading image...")
image = tifffile.imread(input_image_path)

# Draw polygons
print("Drawing polygons...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Parse polygon from JSON string
    if 'polygon' in df.columns:
        polygon_data = json.loads(row['polygon'])
    else:
        print("Available columns:", df.columns.tolist())
        break

    # Convert from nested list format [[[x, y]]] to simple [[x, y]]
    coords = np.array([[point[0][0], point[0][1]] for point in polygon_data], dtype=np.int32)

    # Draw on image
    cv2.polylines(image, [coords], isClosed=True, color=color, thickness=thickness)

# Save as temporary uncompressed TIFF
print("Saving temporary TIFF...")
temp_tiff = tempfile.mktemp(suffix='.tif')
tifffile.imwrite(temp_tiff, image, photometric='rgb')

# Compress with GDAL
print(f"Compressing with JPEG (quality={jpeg_quality})...")
gdal.Translate(
    output_image_path,
    temp_tiff,
    creationOptions=[
        'COMPRESS=JPEG',
        f'JPEG_QUALITY={jpeg_quality}',
        'TILED=YES',
        'PHOTOMETRIC=YCBCR'
    ]
)

# Remove temporary file
os.remove(temp_tiff)

# Add overviews
print(f"Building overviews: {overview_levels}...")
ds = gdal.Open(output_image_path, gdal.GA_Update)
ds.BuildOverviews('AVERAGE', overview_levels)
ds = None

print(f"Output saved to: {output_image_path}")
print("Done!")