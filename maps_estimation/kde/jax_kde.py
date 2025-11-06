import numpy as np
import h5py
import json
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
from time import time

Image.MAX_IMAGE_PIXELS = None


def extract_label2_coords(h5_path):
    """Extract all coordinates for label 2 from h5 file"""
    coords = []
    with h5py.File(h5_path, 'r') as f:
        for key in f['wsi_cells'].keys():
            if key.startswith('tile_'):
                data = f['wsi_cells'][key][()]
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                geojson = json.loads(data)
                for feature in geojson['features']:
                    if feature['properties']['label'] == 2:
                        coords.append(feature['geometry']['coordinates'])
    return np.array(coords)


def estimate_kde(cd8_path, krt_path, output_dir, reference_image_path, bandwidth,
                 tile_level, grid_resolution, map_size):
    start_time = time()
    os.makedirs(output_dir, exist_ok=True)
    # Get reference dimensions
    with tifffile.TiffFile(reference_image_path) as tif:
        page = tif.pages[tile_level]
        ref_height, ref_width = page.shape[:2]

    aspect_ratio = ref_width / ref_height
    output_width = map_size if aspect_ratio >= 1 else int(map_size * aspect_ratio)
    output_height = int(map_size / aspect_ratio) if aspect_ratio >= 1 else map_size
    if map_size == -1:
        output_height = ref_height
        output_width = ref_width

    print(f"Reference: {ref_width}x{ref_height}, Output: {output_width}x{output_height}")

    # Extract coordinates
    coords = extract_label2_coords(cd8_path)
    x, y = coords[:, 0], coords[:, 1]
    print(f"Loaded {len(x)} points")

    # Create KDE
    kde_start = time()
    data_jax = jnp.vstack([x, y])
    kde = gaussian_kde(data_jax, bw_method=bandwidth)
    print(f"KDE created in {time() - kde_start:.3f}s")

    # Evaluate density
    grid_width = int(grid_resolution * aspect_ratio)
    grid_height = grid_resolution

    xx, yy = jnp.meshgrid(
        jnp.linspace(0, ref_width, grid_width),
        jnp.linspace(0, ref_height, grid_height)
    )
    grid_points = jnp.vstack([xx.ravel(), yy.ravel()])

    eval_start = time()
    density = kde(grid_points).reshape(xx.shape)
    density = np.array(density)

    sorted_d = np.sort(density.ravel())
    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    print(f"Density evaluated in {time() - eval_start:.3f}s")

    # Create and save heatmap
    density_norm = (density - density.min()) / (density.max() - density.min())
    import matplotlib.pyplot as plt
    density_rgb = (plt.get_cmap('hot')(density_norm)[:, :, :3] * 255).astype(np.uint8)
    density_resized = Image.fromarray(density_rgb).resize((output_width, output_height), Image.BILINEAR)

    # Save as TIFF
    heatmap_tiff = os.path.join(output_dir, f'bw_{bandwidth}_heatmap.tif')
    temp_tiff = tempfile.mktemp(suffix='.tif')
    tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')
    gdal.Translate(heatmap_tiff, temp_tiff,
                   creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES'])
    os.remove(temp_tiff)

    ds = gdal.Open(heatmap_tiff, gdal.GA_Update)
    ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
    ds = None

    print(f"Total time: {time() - start_time:.3f}s")
    print(f"Saved: {heatmap_tiff}")

    return heatmap_tiff, gini


if __name__ == '__main__':
    folder_path = "/mnt/c/Users/tomer/Desktop/data/project 1/225_panCK CD8_TRSPZ005647_u673_1_40X"
    cd8_path = os.path.join(folder_path, "cd8.hdf5")
    krt_path = os.path.join(folder_path, "krt.hdf5")
    output_dir = os.path.join(folder_path, "output")
    ref_tiff = os.path.join(folder_path, "225_panCK CD8_TRSPZ005647_u673_1_40X.tif")

    results_file = os.path.join(output_dir, "gini_results.txt")

    import csv

    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bandwidth", "Gini"])

        for bw in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
            heatmap_tiff, gini = estimate_kde(
                cd8_path, krt_path, output_dir, ref_tiff,
                bandwidth=bw, tile_level=2, grid_resolution=1500, map_size=1000
            )
            writer.writerow([bw, f"{gini:.6f}"])
