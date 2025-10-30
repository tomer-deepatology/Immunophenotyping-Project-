import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
import json
from matplotlib.path import Path

Image.MAX_IMAGE_PIXELS = None

from time import time


def create_tissue_mask_at_grid(geojson_path, grid_width, grid_height, ref_width, ref_height):
    """
    Create a binary mask from GeoJSON annotations at grid resolution.
    Much faster than creating at full image resolution.
    Returns a boolean array where True = inside tissue region.
    """
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)

    # Create empty mask at grid resolution
    mask = np.zeros((grid_height, grid_width), dtype=bool)

    # Create grid points in image coordinates
    x_coords = np.linspace(0, ref_width, grid_width)
    y_coords = np.linspace(0, ref_height, grid_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Check all grid points against all polygons
    for feature in geojson['features']:
        coords = np.array(feature['geometry']['coordinates'][0])
        path = Path(coords)
        inside = path.contains_points(points)
        mask |= inside.reshape(grid_height, grid_width)

    return mask


def calculate_gini_for_bandwidth(csv_path, bandwidth, reference_tiff_path, geojson_path=None,
                                 tile_level=0, grid_resolution=100):
    """
    Calculate only the Gini coefficient for a given bandwidth.
    No plotting or heatmap saving - just the Gini value.
    """
    # Get dimensions from reference image
    # Handle both TIFF and PNG files
    if reference_tiff_path.lower().endswith('.tif') or reference_tiff_path.lower().endswith('.tiff'):
        with tifffile.TiffFile(reference_tiff_path) as tif:
            page = tif.pages[tile_level]
            ref_height, ref_width = page.shape[:2]
    else:
        # For PNG or other image formats
        img = Image.open(reference_tiff_path)
        ref_width, ref_height = img.size
        img.close()

    aspect_ratio = ref_width / ref_height

    # Load points
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        df = df[df['category'] == 'Category 2']
    x, y = df['x'].values, df['y'].values

    # Create KDE
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

    # Evaluate on grid
    grid_width = int(grid_resolution * aspect_ratio)
    grid_height = grid_resolution

    # Create tissue mask if GeoJSON provided (at grid resolution)
    tissue_mask = None
    if geojson_path is not None and os.path.exists(geojson_path):
        tissue_mask = create_tissue_mask_at_grid(geojson_path, grid_width, grid_height,
                                                 ref_width, ref_height)

    xx, yy = np.meshgrid(
        np.linspace(0, ref_width, grid_width),
        np.linspace(0, ref_height, grid_height)
    )
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Calculate Gini coefficient - MASKED version if tissue mask provided
    if tissue_mask is not None:
        density_masked = density[tissue_mask]
        sorted_d = np.sort(density_masked.ravel())
    else:
        sorted_d = np.sort(density.ravel())

    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    return gini


def process_bandwidth_sweep(paths, bandwidth_range=None, tile_level=2, grid_resolution=100):
    """
    Process a single image for a range of bandwidths and create plots.

    Parameters:
    - paths: Dictionary containing 'folder_dir', 'ref_tiff', 'csv_path', 'geojson_path'
    - bandwidth_range: Array of bandwidth values to test (default: np.linspace(0.001, 0.4, 50))
    - tile_level: TIFF tile level to use
    - grid_resolution: Grid resolution for KDE evaluation
    """
    if bandwidth_range is None:
        bandwidth_range = np.linspace(0.001, 0.4, 50)

    folder_dir = paths['folder_dir']
    ref_tiff = paths['ref_tiff']
    csv_path = paths['csv_path']
    geojson_path = paths.get('geojson_path')  # Use .get() for safer access

    print(f"\nProcessing: {os.path.basename(ref_tiff)}")
    print(f"Bandwidth range: {bandwidth_range[0]:.3f} to {bandwidth_range[-1]:.3f} ({len(bandwidth_range)} steps)")
    if geojson_path:
        print("Using tissue mask: Yes")
    else:
        print("Using tissue mask: No")
    print("=" * 80)

    # Arrays to store results
    gini_masked = []
    gini_unmasked = []

    # Calculate Gini for each bandwidth
    start_time = time()
    for i, bw in enumerate(bandwidth_range):
        print(f"Progress: {i + 1}/{len(bandwidth_range)} - BW={bw:.4f}", end='\r')

        # Calculate masked Gini (only if geojson_path is provided)
        if geojson_path:
            gini_m = calculate_gini_for_bandwidth(csv_path, bw, ref_tiff,
                                                  geojson_path=geojson_path,
                                                  tile_level=tile_level,
                                                  grid_resolution=grid_resolution)
            gini_masked.append(gini_m)

        # Calculate unmasked Gini
        gini_u = calculate_gini_for_bandwidth(csv_path, bw, ref_tiff,
                                              geojson_path=None,
                                              tile_level=tile_level,
                                              grid_resolution=grid_resolution)
        gini_unmasked.append(gini_u)

    elapsed_time = time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f}s ({elapsed_time / len(bandwidth_range):.2f}s per bandwidth)")

    # Create plots - either 1 or 2 subplots depending on whether we have masked data
    if geojson_path and gini_masked:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Masked Gini
        ax1.plot(bandwidth_range, gini_masked, 'b-', linewidth=2)
        ax1.set_xlabel('Bandwidth', fontsize=12)
        ax1.set_ylabel('Gini Coefficient (Masked)', fontsize=12)
        ax1.set_title(f'Gini vs Bandwidth - Tissue Segmented\n{os.path.basename(ref_tiff)}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(bandwidth_range[0], bandwidth_range[-1])
        ax1.set_ylim(0, 1)
        textstr_masked = (f'Min Gini: {min(gini_masked):.4f} (BW={bandwidth_range[np.argmin(gini_masked)]:.4f})\n'
                          f'Max Gini: {max(gini_masked):.4f} (BW={bandwidth_range[np.argmax(gini_masked)]:.4f})')
        ax1.text(0.98, 0.02, textstr_masked, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Plot 2: Unmasked Gini
        ax2.plot(bandwidth_range, gini_unmasked, 'r-', linewidth=2)
        ax2.set_xlabel('Bandwidth', fontsize=12)
        ax2.set_ylabel('Gini Coefficient (Unmasked)', fontsize=12)
        ax2.set_title(f'Gini vs Bandwidth - Full Image\n{os.path.basename(ref_tiff)}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(bandwidth_range[0], bandwidth_range[-1])
        ax2.set_ylim(0, 1)
        textstr_unmasked = (f'Min Gini: {min(gini_unmasked):.4f} (BW={bandwidth_range[np.argmin(gini_unmasked)]:.4f})\n'
                            f'Max Gini: {max(gini_unmasked):.4f} (BW={bandwidth_range[np.argmax(gini_unmasked)]:.4f})')
        ax2.text(0.98, 0.02, textstr_unmasked, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        # Only unmasked plot
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        ax2.plot(bandwidth_range, gini_unmasked, 'r-', linewidth=2)
        ax2.set_xlabel('Bandwidth', fontsize=12)
        ax2.set_ylabel('Gini Coefficient', fontsize=12)
        ax2.set_title(f'Gini vs Bandwidth - Full Image\n{os.path.basename(ref_tiff)}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(bandwidth_range[0], bandwidth_range[-1])
        ax2.set_ylim(0, 1)
        textstr_unmasked = (f'Min Gini: {min(gini_unmasked):.4f} (BW={bandwidth_range[np.argmin(gini_unmasked)]:.4f})\n'
                            f'Max Gini: {max(gini_unmasked):.4f} (BW={bandwidth_range[np.argmax(gini_unmasked)]:.4f})')
        ax2.text(0.98, 0.02, textstr_unmasked, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Save plot
    output_plot_path = os.path.join(folder_dir, "gini_vs_bandwidth_evaluations.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_plot_path}")

    # Save data to CSV
    if geojson_path and gini_masked:
        results_df = pd.DataFrame({
            'bandwidth': bandwidth_range,
            'gini_masked': gini_masked,
            'gini_unmasked': gini_unmasked
        })
    else:
        results_df = pd.DataFrame({
            'bandwidth': bandwidth_range,
            'gini_unmasked': gini_unmasked
        })

    output_csv_path = os.path.join(folder_dir, "gini_vs_bandwidth_evaluations.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved data: {output_csv_path}")

    print("=" * 80)
    return results_df

def main():
    # List of all images to process
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X"
    # images = [
    #     {
    #         'folder_dir': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X",
    #         'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012209_u673_2_40X.tif",
    #         'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #         'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"
    #     },
    # ]
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ005647_u673_1_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ005647_u673_1_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ008500_u673_1_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ008500_u673_1_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012200_u673_2_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012200_u673_2_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ012200_u673_2_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012212_u673_2_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012212_u673_2_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ012212_u673_2_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ015156_u673_1_40X-005"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ015156_u673_1_40X-005.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ015156_u673_1_40X-005.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014171_u673_1_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014171_u673_1_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ014171_u673_1_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014460_u673_1_40X-006"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014460_u673_1_40X-006.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ014460_u673_1_40X-006.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014459_u673_1_40X"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014459_u673_1_40X.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ014459_u673_1_40X.tif - Series 0.geojson"
    # })
    #
    # folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014174_u673_1_40X-001"
    # images.append({
    #     'folder_dir': folder_dir,
    #     'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif",
    #     'csv_path': fr"{folder_dir}\detections_from_iris.csv",
    #     'geojson_path': fr"{folder_dir}\225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif - Series 0.geojson"
    # })
    images = []
    for i in range(3, 8):
        folder_dir = rf"C:\Users\tomer\Desktop\data\project 1\synthetics_data\sync_points_{i}"
        images.append({
            'folder_dir': folder_dir,
            'ref_tiff': fr"{folder_dir}\sync_points_{i}.png",
            'csv_path': fr"{folder_dir}\sync_points_{i}.csv",
            'geojson_path': None

        })
    # Define bandwidth range
    bandwidth_range = np.linspace(0.001, 0.4, 150)

    # Process each image
    for paths in images:
        try:
            process_bandwidth_sweep(paths,
                                    bandwidth_range=bandwidth_range,
                                    tile_level=2,
                                    grid_resolution=100)
        except Exception as e:
            print(f"ERROR processing {paths['ref_tiff']}: {str(e)}")
            print("Continuing to next image...")
            continue

    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
