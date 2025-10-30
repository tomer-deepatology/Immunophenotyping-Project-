import pandas as pd
import tifffile
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import cv2
from tqdm import tqdm

# ===== PARAMETERS =====
csv_path = r"C:\Users\tomer\Desktop\data\demo_sunday\report\2025-10-30_full_detections.csv"
input_image_path = r"C:\Users\tomer\Desktop\data\demo_sunday\demo_best_resolution.tif"
output_image_path = r"C:\Users\tomer\Desktop\data\demo_sunday\detections_overlay.tif"
color = (255, 0, 0)  # Red in RGB
thickness = 2  # Line thickness (-1 for filled)
# ======================

# Load CSV
print("Loading detections...")
df = pd.read_csv(csv_path)
print(f"Total detections: {len(df)}")

# Load image
print("Loading image...")
image = tifffile.imread(input_image_path)

# Draw polygons
print("Drawing polygons...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Parse polygon from WKT or coordinates column
    if 'geometry' in df.columns:
        poly = wkt.loads(row['geometry'])
    elif 'Polygon' in df.columns:
        poly = wkt.loads(row['Polygon'])
    else:
        # Assuming there's a column with polygon data
        print("Available columns:", df.columns.tolist())
        break

    # Get coordinates
    coords = np.array(poly.exterior.coords, dtype=np.int32)

    # Draw on image
    cv2.polylines(image, [coords], isClosed=True, color=color, thickness=thickness)

# Save output
print("Saving image...")
tifffile.imwrite(output_image_path, image)

print(f"Output saved to: {output_image_path}")