import tifffile
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.patches import Polygon

# Paths
im_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
annotations_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"
output_tiff_path  =r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X\cropped_with_annotations.tif"
# Load TIFF and annotations
img = tifffile.imread(im_path)
with open(annotations_path, 'r') as f:
    geojson = json.load(f)

# Get bounding box of all polygons
all_coords = []
for feature in geojson['features']:
    all_coords.extend(feature['geometry']['coordinates'][0])
all_coords = np.array(all_coords)

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
for feature in geojson['features']:
    coords = np.array(feature['geometry']['coordinates'][0])
    coords[:, 0] -= x_min
    coords[:, 1] -= y_min
    poly = Polygon(coords, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(poly)

ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig(output_tiff_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved cropped image with annotations!")