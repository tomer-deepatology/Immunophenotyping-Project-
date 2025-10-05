import pandas as pd
import json
import numpy as np
from shapely.geometry import Point, Polygon

# Load data
csv_path = r"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\2025-09-29_full_detections.csv"
annotations_path = r"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"

df = pd.read_csv(csv_path)

with open(annotations_path, 'r') as f:
    geojson = json.load(f)

# Get bounding box of all polygons
all_coords = []
for feature in geojson['features']:
    all_coords.extend(feature['geometry']['coordinates'][0])
all_coords = np.array(all_coords)

x_min, y_min = all_coords.min(axis=0)
x_max, y_max = all_coords.max(axis=0)

# Add margin to bounding box
bbox_margin = 100
x_min -= bbox_margin
x_max += bbox_margin
y_min -= bbox_margin
y_max += bbox_margin

# Filter points within bounding box
df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]

# Create Shapely polygons with buffer
poly_margin = 20  # negative shrinks, positive expands
polygons = [Polygon(feature['geometry']['coordinates'][0]).buffer(poly_margin) for feature in geojson['features']]

# Check if each point is inside any polygon
def point_in_polygons(x, y, polygons):
    point = Point(x, y)
    return any(poly.contains(point) for poly in polygons)

df['inside_krt'] = df.apply(lambda row: point_in_polygons(row['x'], row['y'], polygons), axis=1)

# Save to new CSV
output_path = r"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\2025-09-29_filtered_detections_with_krt.csv"
df.to_csv(output_path, index=False)

print(f"Total points in bbox: {len(df)}")
print(f"Inside KRT: {df['inside_krt'].sum()}")
print(f"Outside KRT: {(~df['inside_krt']).sum()}")
print(f"Saved: {output_path}")