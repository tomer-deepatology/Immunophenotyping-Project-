import h5py
import numpy as np
import csv
import json
from pathlib import Path

annotations_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X_report\c6e89ee9-3ce1-4b69-b3a6-953e950fbec5\mask_file.hdf5"

# Automatically determine output path
output_csv = Path(annotations_path).parent.parent / "detections_from_iris.csv"

with h5py.File(annotations_path, 'r') as f:
    # Get all tile keys
    tile_keys = [k for k in f['wsi_cells'].keys() if k.startswith('tile_')]

    print(f"Found {len(tile_keys)} tiles\n")

    # Extract cells from all tiles
    all_cells = []
    for tile_key in tile_keys:
        tile_data = f[f'wsi_cells/{tile_key}'][()]
        cells = json.loads(tile_data)

        for feature in cells['features']:
            x, y = feature['geometry']['coordinates']
            label = feature['properties']['label']
            all_cells.append({'x': x, 'y': y, 'label': f'Category {label}'})

    print(f"Total cells: {len(all_cells)}\n")

    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['x', 'y', 'label'])
        writer.writeheader()
        writer.writerows(all_cells)

    print(f"\nSaved {len(all_cells)} cells to {output_csv}")