import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
from time import time
import openslide

Image.MAX_IMAGE_PIXELS = None


def estimate_kde(csv_path, folder_dir, bandwidth, reference_image_path, tile_level=0, grid_resolution=100,
                 map_size=1000, save_heatmap=True, save_plot=True):
    os.makedirs(folder_dir, exist_ok=True)
    init_time = time()

    # Get image dimensions
    file_ext = os.path.splitext(reference_image_path)[1].lower()
    if file_ext in ['.tif', '.tiff']:
        with tifffile.TiffFile(reference_image_path) as tif:
            page = tif.pages[tile_level]
            ref_height, ref_width = page.shape[:2]
    elif file_ext == '.ndpi':
        slide = openslide.OpenSlide(reference_image_path)
        ref_width, ref_height = slide.level_dimensions[tile_level]
        slide.close()
    else:
        img = Image.open(reference_image_path)
        ref_width, ref_height = img.size
        img.close()

    aspect_ratio = ref_width / ref_height

    # Handle map_size: -1 or negative means use original dimensions
    if map_size <= 0:
        output_width = ref_width
        output_height = ref_height
    else:
        output_width = map_size if aspect_ratio >= 1 else int(map_size * aspect_ratio)
        output_height = int(map_size / aspect_ratio) if aspect_ratio >= 1 else map_size

    print(f"Reference image: {ref_width}x{ref_height}")
    print(f"Output map size: {output_width}x{output_height}")

    # Load points
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        df = df[df['category'] == 'Category 2']
    x, y = df['x'].values, df['y'].values
    print(f"Loaded {len(x)} points")

    # Convert relative bandwidth to absolute using scipy
    data = np.vstack([x, y])
    kde_temp = gaussian_kde(data, bw_method=bandwidth)
    absolute_bw = kde_temp.factor * np.std(data)
    print(f"Bandwidth: {bandwidth} (relative) → {absolute_bw:.2f} pixels (absolute)")

    # Use FFTKDE with absolute bandwidth
    kde = FFTKDE(kernel='gaussian', bw=absolute_bw)
    kde.fit(data.T)
    kde_time = time() - init_time
    print(f"KDE creation took: {kde_time:.3f}s")

    # Evaluate on grid
    reference_size = 10000  # pixels (baseline for grid_resolution parameter)
    actual_size = np.mean([ref_width, ref_height])
    scaled_grid = int(grid_resolution * (actual_size / reference_size))
    print(f"Grid resolution: {scaled_grid}x{scaled_grid} ({scaled_grid ** 2:,} total points)")

    grid, points = kde.evaluate(scaled_grid)

    x_grid = np.unique(grid[:, 0])
    y_grid = np.unique(grid[:, 1])
    density = points.reshape(len(y_grid), len(x_grid)).T

    density_time = time() - init_time - kde_time
    print(f"Density evaluation took: {density_time:.3f}s")

    # Calculate Gini
    sorted_d = np.sort(density.ravel())
    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    # Save heatmap if requested
    if save_heatmap:
        from scipy.interpolate import griddata
        # Create target regular grid
        xx, yy = np.meshgrid(np.linspace(0, ref_width, scaled_grid),
                             np.linspace(0, ref_height, scaled_grid))
        # Interpolate scattered points onto regular grid
        density = griddata(grid, points, (xx, yy), method='linear')

        density_norm = (density - density.min()) / (density.max() - density.min())
        density_rgb = (plt.get_cmap('hot')(density_norm)[:, :, :3] * 255).astype(np.uint8)
        density_resized = Image.fromarray(density_rgb).resize((output_width, output_height), Image.BILINEAR)
        heatmap_tiff = rf'{folder_dir}\fftdke_bw_{bandwidth}_heatmap.tif'
        temp_tiff = tempfile.mktemp(suffix='.tif')
        tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')
        gdal.Translate(heatmap_tiff, temp_tiff,
                       creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=90', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
        os.remove(temp_tiff)

        ds = gdal.Open(heatmap_tiff, gdal.GA_Update)
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None
        print(f"Saved heatmap: {heatmap_tiff}")

    # Create plot if requested
    if save_plot:
        fig_width = 12
        fig_height = fig_width / aspect_ratio
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(density, extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                   cmap='hot', aspect='equal', origin='upper', interpolation='bilinear')
        plt.colorbar(label='Density', fraction=0.046, pad=0.04)
        plt.scatter(x, y, c='cyan', s=1, alpha=0.3, edgecolors='none')
        plt.title(f'KDE Distribution (BW={bandwidth}), Gini: {gini:.4f}')

        textstr = f'Bandwidth: {bandwidth}\nGini: {gini:.4f}'
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plot_path = rf'{folder_dir}\fftdke_bw_{bandwidth}_heatmap_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"Total runtime: {time() - init_time:.3f}s")
    return {'gini': gini, 'n_points': len(x)}


def main():
    folder_dir = r"C:\Users\tomer\Desktop\data\demo_sunday"
    ref_tiff = fr"{folder_dir}\OS-2.ndpi"
    csv_path = fr"{folder_dir}\report\2025-10-30_full_detections.csv"

    # Example: use original image dimensions for output
    results = estimate_kde(csv_path,
                           folder_dir,
                           0.1,
                           ref_tiff,
                           tile_level=0,
                           grid_resolution=100,
                           map_size=1000,  # Use -1 or any negative number for original dimensions
                           save_heatmap=True,
                           save_plot=False)

    exit()
    # for bw in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]:
    for bw in [0.5]:
        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X"
        images = [
            {
                'folder_dir': rf'{folder_dir}\fft_kde',
                'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012209_u673_2_40X.tif",
                'csv_path': fr"{folder_dir}\detections_from_iris.csv",
            },
        ]

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ005647_u673_1_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ008500_u673_1_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012200_u673_2_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012200_u673_2_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012212_u673_2_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ012212_u673_2_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ015156_u673_1_40X-005"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ015156_u673_1_40X-005.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014171_u673_1_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014171_u673_1_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014460_u673_1_40X-006"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014460_u673_1_40X-006.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014459_u673_1_40X"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014459_u673_1_40X.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014174_u673_1_40X-001"
        images.append({
            'folder_dir': rf'{folder_dir}\fft_kde',
            'ref_tiff': fr"{folder_dir}\225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif",
            'csv_path': fr"{folder_dir}\detections_from_iris.csv",
        })

        for idx, sample in enumerate(images):
            print("\n" + "=" * 80)
            print(f"PROCESSING SAMPLE {idx}/{len(images)}")
            print("=" * 80)

            folder_dir = sample['folder_dir']
            ref_tiff = sample['ref_tiff']
            csv_path = sample['csv_path']

            results = estimate_kde(csv_path,
                                   folder_dir,
                                   bw,
                                   ref_tiff,
                                   tile_level=2,
                                   grid_resolution=100,
                                   map_size=-1,  # Use -1 or any negative number for original dimensions
                                   # map_size=500,  # Use -1 or any negative number for original dimensions
                                   save_heatmap=True,
                                   save_plot=False)




if __name__ == '__main__':
    main()