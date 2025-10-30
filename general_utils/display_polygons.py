import tifffile
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import ast

# Paths
# im_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
# annotations_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"
# output_tiff_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\cropped_with_annotations_23.tif"

im_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\225_panCK CD8_TRSPZ005647_u673_1_40X.tif"
annotations_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\report\2025-10-26_full_detections.csv"
output_tiff_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\test.tif"


def load_polygons_from_geojson(geojson_path):
    """Load polygons from GeoJSON format"""
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)

    polygons = []
    for feature in geojson['features']:
        coords = np.array(feature['geometry']['coordinates'][0])
        polygons.append(coords)

    return polygons


def load_polygons_from_csv(csv_path, slide_filter=None):
    """Load polygons from CSV format

    Args:
        csv_path: Path to CSV file
        slide_filter: Optional slide name to filter polygons (e.g., '225_panCK CD8_TRSPZ012209_u673_2_40X')
    """
    df = pd.read_csv(csv_path)

    # Filter by slide if specified
    if slide_filter:
        df = df[df['slide'] == slide_filter]

    polygons = []
    for polygon_str in df['polygon']:
        # Parse the string representation of the polygon
        # The format is: [[[x1, y1]], [[x2, y2]], ...]
        polygon_list = ast.literal_eval(polygon_str)

        # Convert from [[[x, y]], [[x, y]], ...] to [[x, y], [x, y], ...]
        coords = np.array([[point[0][0], point[0][1]] for point in polygon_list])
        polygons.append(coords)

    return polygons


# Load TIFF image
img = tifffile.imread(im_path)

# Extract slide name from TIFF filename (for CSV filtering)
import os

slide_name = os.path.splitext(os.path.basename(im_path))[0]

# Auto-detect format from file extension
if annotations_path.lower().endswith('.geojson') or annotations_path.lower().endswith('.json'):
    polygons = load_polygons_from_geojson(annotations_path)
    print(f"Loaded {len(polygons)} polygons from GeoJSON file")
elif annotations_path.lower().endswith('.csv'):
    polygons = load_polygons_from_csv(annotations_path, slide_filter=slide_name)
    print(f"Loaded {len(polygons)} polygons from CSV file (filtered by slide: {slide_name})")
else:
    raise ValueError(f"Unsupported file format. Use .geojson, .json, or .csv. Got: {annotations_path}")

# Get bounding box of all polygons
all_coords = np.vstack(polygons)
x_min, y_min = all_coords.min(axis=0)
x_max, y_max = all_coords.max(axis=0)

# Add margin
margin = 0
x_min = max(0, int(x_min - margin))
x_max = min(img.shape[1], int(x_max + margin))
y_min = max(0, int(y_min - margin))
y_max = min(img.shape[0], int(y_max + margin))

# Crop image
cropped_img = img[y_min:y_max, x_min:x_max]

# Plot
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(cropped_img)

# Draw polygons (adjust coordinates for crop)
for coords in polygons:
    # Adjust coordinates relative to the crop
    adjusted_coords = coords.copy()
    adjusted_coords[:, 0] -= x_min
    adjusted_coords[:, 1] -= y_min

    poly = Polygon(adjusted_coords, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(poly)

ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig(output_tiff_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Saved cropped image with {len(polygons)} annotations to {output_tiff_path}")