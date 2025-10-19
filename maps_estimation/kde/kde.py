import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os

Image.MAX_IMAGE_PIXELS = None

from time import time


def estimate_kde(csv_path, folder_dir, bandwidth, reference_tiff_path, tile_level=0, grid_resolution=100,
                 map_size=1000):
    init_time = time()

    # Get dimensions from specific TIFF tile/page
    with tifffile.TiffFile(reference_tiff_path) as tif:
        print(f"Total pages in TIFF: {len(tif.pages)}")
        page = tif.pages[tile_level]
        ref_height, ref_width = page.shape[:2]
        print(f"Reference tile {tile_level}: {ref_width}x{ref_height}")

    aspect_ratio = ref_width / ref_height
    print(f"Aspect ratio: {aspect_ratio:.4f}")

    # Calculate output dimensions based on aspect ratio
    if aspect_ratio >= 1:
        # Width is larger, so map_size is the width
        output_width = map_size
        output_height = int(map_size / aspect_ratio)
    else:
        # Height is larger, so map_size is the height
        output_height = map_size
        output_width = int(map_size * aspect_ratio)

    print(f"Output map size: {output_width}x{output_height}")

    # Load points
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        df = df[df['category'] == 'Category 2']
    x, y = df['x'].values, df['y'].values
    print(f"Loaded {len(x)} points")

    # Create KDE
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    kde_time = time() - init_time
    print(f"KDE creation took: {kde_time:.3f}s")

    # Evaluate on grid with same resolution in both dimensions
    # Calculate grid points based on aspect ratio to maintain same resolution
    grid_width = int(grid_resolution * aspect_ratio)
    grid_height = grid_resolution

    print(f"Grid size: {grid_width}x{grid_height}")

    xx, yy = np.meshgrid(
        np.linspace(0, ref_width, grid_width),
        np.linspace(0, ref_height, grid_height)
    )
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density_time = time() - init_time - kde_time
    print(f"Density evaluation took: {density_time:.3f}s")

    # Calculate metrics
    sorted_d = np.sort(density.ravel())
    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    # Create heatmap image
    density_norm_img = (density - density.min()) / (density.max() - density.min())
    density_rgb = (plt.get_cmap('hot')(density_norm_img)[:, :, :3] * 255).astype(np.uint8)
    density_resized = Image.fromarray(density_rgb).resize((output_width, output_height), Image.BILINEAR)

    # Save heatmap as TIFF with JPEG compression
    heatmap_tiff = rf'{folder_dir}\bw_{bandwidth}_heatmap.tif'
    temp_tiff = tempfile.mktemp(suffix='.tif')

    tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')

    gdal.Translate(heatmap_tiff, temp_tiff,
                   creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
    os.remove(temp_tiff)

    # Add overviews
    ds = gdal.Open(heatmap_tiff, gdal.GA_Update)
    ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
    ds = None
    print(f"Saved: {heatmap_tiff}")

    # Save plot with metrics - using reference image dimensions for correct aspect ratio
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    plt.figure(figsize=(fig_width, fig_height))

    # Use reference image dimensions for extent to maintain correct aspect ratio
    # Added interpolation='bilinear' for smooth appearance
    plt.imshow(density, extent=[0, ref_width, ref_height, 0], cmap='hot', aspect='equal',
               origin='upper', interpolation='bilinear')
    plt.colorbar(label='Density', fraction=0.046, pad=0.04)
    plt.scatter(x, y, c='cyan', s=1, alpha=0.3, edgecolors='none')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'KDE Distribution (BW={bandwidth}), Gini: {gini:.4f}')

    # Set axis limits to reference image dimensions
    plt.xlim(0, ref_width)
    plt.ylim(ref_height, 0)

    textstr = (f'Bandwidth: {bandwidth}\n'
               f'Gini Coefficient: {gini:.4f}\n'
               f'(0=uniform, 1=concentrated)')

    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             family='monospace')

    plot_path = rf'{folder_dir}\bw_{bandwidth}_heatmap_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    total_time = time() - init_time
    print(f"Total runtime: {total_time:.3f}s")

    return {'gini': gini, 'n_points': len(x)}


def main():
    folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X"
    ref_tiff = fr"{folder_dir}\225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
    csv_path = fr"{folder_dir}\detections_from_iris.csv"
    bw = 0.05

    # map_size controls the larger dimension (width or height depending on aspect ratio)
    results = estimate_kde(csv_path, folder_dir, bw, ref_tiff,
                           tile_level=2, grid_resolution=100, map_size=2000)
    print(f"\nResults: {results}")


if __name__ == '__main__':
    main()