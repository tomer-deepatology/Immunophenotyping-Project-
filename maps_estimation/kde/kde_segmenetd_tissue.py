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


def estimate_kde(csv_path, heatmap_tiff_path, plot_path, bandwidth, reference_tiff_path, geojson_path=None,
                 tile_level=0, grid_resolution=100, map_size=1000):
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
        output_width = map_size
        output_height = int(map_size / aspect_ratio)
    else:
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

    # Evaluate on grid
    grid_width = int(grid_resolution * aspect_ratio)
    grid_height = grid_resolution
    print(f"Grid size: {grid_width}x{grid_height}")

    # Create tissue mask if GeoJSON provided (at grid resolution, not full resolution!)
    tissue_mask = None
    if geojson_path:
        print("Creating tissue mask at grid resolution...")
        mask_start = time()
        tissue_mask = create_tissue_mask_at_grid(geojson_path, grid_width, grid_height,
                                                 ref_width, ref_height)
        mask_time = time() - mask_start
        tissue_area = tissue_mask.sum()
        total_area = grid_width * grid_height
        print(f"Tissue mask created in {mask_time:.3f}s")
        print(f"Tissue area: {tissue_area:,} grid cells ({100 * tissue_area / total_area:.2f}% of grid)")

    xx, yy = np.meshgrid(
        np.linspace(0, ref_width, grid_width),
        np.linspace(0, ref_height, grid_height)
    )
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density_time = time() - init_time - kde_time
    print(f"Density evaluation took: {density_time:.3f}s")

    # Calculate Gini coefficient - MASKED version if tissue mask provided
    if tissue_mask is not None:
        # Mask is already at grid resolution, so use directly!
        density_masked = density[tissue_mask]
        print(f"Masked density values: {len(density_masked):,} (from {density.size:,} total grid points)")

        # Calculate Gini on masked density
        sorted_d = np.sort(density_masked.ravel())
    else:
        # Original unmasked calculation
        sorted_d = np.sort(density.ravel())

    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    # Save heatmap as TIFF only if path is provided
    if heatmap_tiff_path is not None:
        # Create heatmap image
        density_norm_img = (density - density.min()) / (density.max() - density.min())
        density_rgb = (plt.get_cmap('hot')(density_norm_img)[:, :, :3] * 255).astype(np.uint8)
        density_resized = Image.fromarray(density_rgb).resize((output_width, output_height), Image.BILINEAR)

        temp_tiff = tempfile.mktemp(suffix='.tif')

        tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')

        gdal.Translate(heatmap_tiff_path, temp_tiff,
                       creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
        os.remove(temp_tiff)

        # Add overviews
        ds = gdal.Open(heatmap_tiff_path, gdal.GA_Update)
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None
        print(f"Saved: {heatmap_tiff_path}")
    else:
        print("Heatmap TIFF path is None, skipping heatmap save")

    # Save plot with metrics only if path is provided
    if plot_path is not None:
        fig_width = 12
        fig_height = fig_width / aspect_ratio
        plt.figure(figsize=(fig_width, fig_height))

        plt.imshow(density, extent=[0, ref_width, ref_height, 0], cmap='hot', aspect='equal',
                   origin='upper', interpolation='bilinear')
        plt.colorbar(label='Density', fraction=0.046, pad=0.04)
        plt.scatter(x, y, c='cyan', s=1, alpha=0.3, edgecolors='none')

        # Overlay tissue mask outline if available
        if tissue_mask is not None and geojson_path:
            with open(geojson_path, 'r') as f:
                geojson = json.load(f)
            for feature in geojson['features']:
                coords = np.array(feature['geometry']['coordinates'][0])
                plt.plot(coords[:, 0], coords[:, 1], 'lime', linewidth=1.5, alpha=0.7)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'KDE Distribution (BW={bandwidth}), Gini{"(Masked)" if tissue_mask is not None else ""}: {gini:.4f}')

        plt.xlim(0, ref_width)
        plt.ylim(ref_height, 0)

        textstr = (f'Bandwidth: {bandwidth}\n'
                   f'Gini Coefficient{"(Masked)" if tissue_mask is not None else ""}: {gini:.4f}\n'
                   f'(0=uniform, 1=concentrated)')

        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 family='monospace')

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    else:
        print("Plot path is None, skipping plot save")

    total_time = time() - init_time
    print(f"Total runtime: {total_time:.3f}s")

    return {'gini': gini, 'n_points': len(x), 'masked': tissue_mask is not None}


def main():
    # Define all samples
    samples = [
        {
            'name': 'TRSPZ012209_u673_2',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X",
            'tiff': "225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
        },
        {
            'name': 'TRSPZ005647_u673_1',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X",
            'tiff': "225_panCK CD8_TRSPZ005647_u673_1_40X.tif"
        },
        {
            'name': 'TRSPZ008500_u673_1',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X",
            'tiff': "225_panCK CD8_TRSPZ008500_u673_1_40X.tif"
        },
        {
            'name': 'TRSPZ012200_u673_2',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012200_u673_2_40X",
            'tiff': "225_panCK CD8_TRSPZ012200_u673_2_40X.tif"
        },
        {
            'name': 'TRSPZ012212_u673_2',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012212_u673_2_40X",
            'tiff': "225_panCK CD8_TRSPZ012212_u673_2_40X.tif"
        },
        {
            'name': 'TRSPZ015156_u673_1-005',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ015156_u673_1_40X-005",
            'tiff': "225_panCK CD8_TRSPZ015156_u673_1_40X-005.tif"
        },
        {
            'name': 'TRSPZ014171_u673_1',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014171_u673_1_40X",
            'tiff': "225_panCK CD8_TRSPZ014171_u673_1_40X.tif"
        },
        {
            'name': 'TRSPZ014460_u673_1-006',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014460_u673_1_40X-006",
            'tiff': "225_panCK CD8_TRSPZ014460_u673_1_40X-006.tif"
        },
        {
            'name': 'TRSPZ014459_u673_1',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014459_u673_1_40X",
            'tiff': "225_panCK CD8_TRSPZ014459_u673_1_40X.tif"
        },
        {
            'name': 'TRSPZ014174_u673_1-001',
            'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014174_u673_1_40X-001",
            'tiff': "225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif"
        }
    ]

    bw = 0.001

    # Process each sample
    for i, sample in enumerate(samples, 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING SAMPLE {i}/{len(samples)}: {sample['name']}")
        print("=" * 80)

        folder_dir = sample['folder']
        ref_tiff = os.path.join(folder_dir, sample['tiff'])
        csv_path = os.path.join(folder_dir, "detections_from_iris.csv")
        geojson_path = os.path.join(folder_dir, f"{sample['tiff']} - Series 0.geojson")

        # Create output paths
        heatmap_masked = os.path.join(folder_dir, f"bw_{bw}_heatmap_segmented.tif")
        plot_masked = os.path.join(folder_dir, f"bw_{bw}_heatmap_plot_segmented.png")
        heatmap_unmasked = os.path.join(folder_dir, f"bw_{bw}_heatmap.tif")
        plot_unmasked = os.path.join(folder_dir, f"bw_{bw}_heatmap_plot.png")

        # Set to None if you don't want to save heatmap TIFFs
        heatmap_masked = None
        heatmap_unmasked = None

        try:
            # WITH tissue mask
            print("\n--- WITH TISSUE MASK ---")
            results_masked = estimate_kde(csv_path, heatmap_masked, plot_masked, bw, ref_tiff,
                                          geojson_path=geojson_path,
                                          tile_level=2, grid_resolution=100, map_size=2000)

            # WITHOUT tissue mask
            print("\n--- WITHOUT TISSUE MASK ---")
            results_unmasked = estimate_kde(csv_path, heatmap_unmasked, plot_unmasked, bw, ref_tiff,
                                            geojson_path=None,
                                            tile_level=2, grid_resolution=100, map_size=2000)

            print(f"\n--- RESULTS FOR {sample['name']} ---")
            print(f"Gini (Masked):   {results_masked['gini']:.4f}")
            print(f"Gini (Unmasked): {results_unmasked['gini']:.4f}")
            print(f"Difference:      {results_masked['gini'] - results_unmasked['gini']:.4f}")

        except Exception as e:
            print(f"\nERROR processing {sample['name']}: {str(e)}")

if __name__ == '__main__':
    main()