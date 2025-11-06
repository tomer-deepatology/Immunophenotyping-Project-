import numpy as np
import h5py
import json
from scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
from time import time

Image.MAX_IMAGE_PIXELS = None

def set_bandwidth_manual(kde, bandwidth_pixels):
    """Set bandwidth directly in pixel units"""
    d = kde.d  # number of dimensions
    kde.covariance = np.eye(d) * (bandwidth_pixels ** 2)
    kde.cho_cov = np.linalg.cholesky(kde.covariance).astype(np.float64)
    kde.log_det = 2*np.log(np.diag(kde.cho_cov * np.sqrt(2*np.pi))).sum()

def combine_heatmaps(krt_density_raw, nonkrt_density_raw, output_dir, bandwidth, gini_krt, gini_nonkrt,output_width, output_height):
    """Combine KRT and non-KRT heatmaps using weighted blending"""
    # Create combined folder
    combined_output_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_output_dir, exist_ok=True)

    # Normalize densities
    krt_norm = (krt_density_raw - krt_density_raw.min()) / (krt_density_raw.max() - krt_density_raw.min())
    non_krt_norm = (nonkrt_density_raw - nonkrt_density_raw.min()) / (
                nonkrt_density_raw.max() - nonkrt_density_raw.min())

    # Create colored heatmaps
    import matplotlib.pyplot as plt
    krt_colored = plt.get_cmap('hot')(krt_norm)[:, :, :3]
    krt_colored[krt_norm <= np.quantile(krt_norm, 0.1)] = [0, 0, 0]

    non_krt_colored = plt.get_cmap('gist_earth')(non_krt_norm)[:, :, :3]
    non_krt_colored[non_krt_norm <= np.quantile(non_krt_norm, 0.1)] = [0, 0, 0]

    # Weighted blending based on KRT density
    krt_weight = (krt_norm ** 0.1)[:, :, np.newaxis]
    combined_colored = krt_weight * krt_colored + (1 - krt_weight) * non_krt_colored
    combined_array = (combined_colored * 255).astype(np.uint8)

    combined_array = Image.fromarray(combined_array).resize((output_width, output_height), Image.BILINEAR)
    combined_array = np.array(combined_array)
    # Save combined heatmap
    combined_tiff = os.path.join(combined_output_dir,
                                 f'bw_{bandwidth}_gini_krt_{gini_krt:.4f}_nonkrt_{gini_nonkrt:.4f}_heatmap_combined.tif')
    temp_tiff = tempfile.mktemp(suffix='.tif')
    tifffile.imwrite(temp_tiff, combined_array, photometric='rgb')
    gdal.Translate(combined_tiff, temp_tiff,
                   creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES'])
    os.remove(temp_tiff)

    ds = gdal.Open(combined_tiff, gdal.GA_Update)
    if ds is not None:
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None

    print(f"Combined: Saved: {combined_tiff}")
    return combined_tiff

def load_krt_mask(krt_path):
    """Load the KRT segmentation mask from h5 file"""
    with h5py.File(krt_path, 'r') as f:
        mask = f['wsi_masks/predicted_region_mask_l0'][:]
        # Load the presentation metadata to identify label values
        presentation_data = json.loads(f['wsi_presentation/masks'][()])
        labels = presentation_data[0]['data']

        # Find KRT label value
        krt_label = None
        tissue_label = None
        for label_info in labels:
            name = label_info['textgui'].lower()
            if 'krt' in name or 'ker' in name:
                krt_label = label_info['label']
            elif 'tissue' in name or 'non' in name:
                tissue_label = label_info['label']

        print(f"KRT label: {krt_label}, Tissue/non-KRT label: {tissue_label}")
        return mask, krt_label, tissue_label


def extract_label2_coords(h5_path):
    """Extract all coordinates for label 2 (CD8) from h5 file"""
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


def separate_coords_by_tissue(coords, mask, krt_label, tissue_label):
    """Separate coordinates into KRT and non-KRT regions"""
    krt_coords = []
    nonkrt_coords = []

    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        # Check bounds
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            label_value = mask[y, x]
            if label_value == krt_label:
                krt_coords.append(coord)
            elif label_value == tissue_label:
                nonkrt_coords.append(coord)

    return np.array(krt_coords), np.array(nonkrt_coords)


def estimate_kde_for_tissue(coords,image_resulotion, tissue_type, output_dir,
                            reference_image_path, bandwidth, tile_level,
                            grid_resolution, map_size):
    """Run KDE for a specific tissue type"""
    start_time = time()
    tissue_output_dir = os.path.join(output_dir, tissue_type)
    os.makedirs(tissue_output_dir, exist_ok=True)

    # Get reference dimensions
    with tifffile.TiffFile(reference_image_path) as tif:
        page = tif.pages[tile_level]
        ref_height, ref_width = page.shape[:2]

    aspect_ratio = ref_width / ref_height
    output_width = map_size if aspect_ratio >= 1 else int(map_size * aspect_ratio)
    output_height = int(map_size / aspect_ratio) if aspect_ratio >= 1 else map_size

    output_width = int(grid_resolution * aspect_ratio)
    output_height = grid_resolution

    if map_size == -1:
        output_height = ref_height
        output_width = ref_width

    print(f"\n{tissue_type}: Reference: {ref_width}x{ref_height}, Output: {output_width}x{output_height}")

    x, y = coords[:, 0], coords[:, 1]
    print(f"{tissue_type}: Loaded {len(x)} points")

    if len(x) < 2:
        print(f"{tissue_type}: Not enough points for KDE")
        return None, None

    # Create KDE
    kde_start = time()
    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    set_bandwidth_manual(kde, bandwidth_pixels=bandwidth / image_resulotion)  # 50 pixels in both x and y
    print(f"{tissue_type}: KDE created in {time() - kde_start:.3f}s")

    # Evaluate density
    grid_width = int(grid_resolution * aspect_ratio)
    grid_height = grid_resolution

    xx, yy = np.meshgrid(
        np.linspace(0, ref_width, grid_width),
        np.linspace(0, ref_height, grid_height)
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    eval_start = time()
    density = kde(grid_points).reshape(xx.shape)

    # Calculate Gini coefficient
    sorted_d = np.sort(density.ravel())
    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    print(f"{tissue_type}: Density evaluated in {time() - eval_start:.3f}s, Gini: {gini:.6f}")

    density_raw = density.copy()

    # Create and save heatmap
    density_norm = (density - density.min()) / (density.max() - density.min())
    import matplotlib.pyplot as plt
    density_rgb = (plt.get_cmap('hot')(density_norm)[:, :, :3] * 255).astype(np.uint8)
    density_resized = Image.fromarray(density_rgb).resize((output_width, output_height), Image.BILINEAR)

    # Save as TIFF with Gini in filename
    heatmap_tiff = os.path.join(tissue_output_dir, f'bw_{bandwidth}_gini_{gini:.4f}_heatmap_{tissue_type}.tif')

    temp_tiff = tempfile.mktemp(suffix='.tif')
    tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')
    gdal.Translate(heatmap_tiff, temp_tiff,
                   creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES'])
    os.remove(temp_tiff)

    ds = gdal.Open(heatmap_tiff, gdal.GA_Update)
    ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
    ds = None

    print(f"{tissue_type}: Total time: {time() - start_time:.3f}s")
    print(f"{tissue_type}: Saved: {heatmap_tiff}")

    return heatmap_tiff, gini, density_resized, density_raw




def estimate_kde(krt_coords, nonkrt_coords,image_resulotion, output_dir, reference_image_path, bandwidth, tile_level, grid_resolution,
                 map_size):
    """Main function to run KDE for both tissue types"""
    os.makedirs(output_dir, exist_ok=True)

    krt_density = None
    nonkrt_density = None

    results = {}

    # Run KDE for KRT regions
    if len(krt_coords) >= 2:
        heatmap_krt, gini_krt, krt_density, krt_density_raw = estimate_kde_for_tissue(
            krt_coords,image_resulotion, "krt", output_dir,
            reference_image_path, bandwidth, tile_level, grid_resolution, map_size
        )
        results['krt'] = {'heatmap': heatmap_krt, 'gini': gini_krt, 'n_points': len(krt_coords)}
    else:
        print("Not enough KRT points for KDE")
        results['krt'] = {'heatmap': None, 'gini': None, 'n_points': len(krt_coords)}

    # Run KDE for non-KRT regions
    if len(nonkrt_coords) >= 2:
        heatmap_nonkrt, gini_nonkrt, nonkrt_density, nonkrt_density_raw = estimate_kde_for_tissue(
            nonkrt_coords,image_resulotion, "nonkrt", output_dir,
            reference_image_path, bandwidth, tile_level, grid_resolution, map_size
        )

        results['nonkrt'] = {'heatmap': heatmap_nonkrt, 'gini': gini_nonkrt, 'n_points': len(nonkrt_coords)}
    else:
        print("Not enough non-KRT points for KDE")
        results['nonkrt'] = {'heatmap': None, 'gini': None, 'n_points': len(nonkrt_coords)}

    if krt_density is not None and nonkrt_density is not None:
        combined_tiff = combine_heatmaps(krt_density_raw, nonkrt_density_raw, output_dir,
                                         bandwidth, results['krt']['gini'], results['nonkrt']['gini'],nonkrt_density.size[0], nonkrt_density.size[1])
        results['combined'] = {'heatmap': combined_tiff}
    else:
        results['combined'] = {'heatmap': None}

    return results


samples = [
    {
        'name': 'TRSPZ012209_u673_2',
        'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012209_u673_2_40X",
        'tiff': "225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
    },
    # {
    #     'name': 'TRSPZ005647_u673_1',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X",
    #     'tiff': "225_panCK CD8_TRSPZ005647_u673_1_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ008500_u673_1',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X",
    #     'tiff': "225_panCK CD8_TRSPZ008500_u673_1_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ012200_u673_2',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012200_u673_2_40X",
    #     'tiff': "225_panCK CD8_TRSPZ012200_u673_2_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ012212_u673_2',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ012212_u673_2_40X",
    #     'tiff': "225_panCK CD8_TRSPZ012212_u673_2_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ015156_u673_1-005',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ015156_u673_1_40X-005",
    #     'tiff': "225_panCK CD8_TRSPZ015156_u673_1_40X-005.tif"
    # },
    # {
    #     'name': 'TRSPZ014171_u673_1',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014171_u673_1_40X",
    #     'tiff': "225_panCK CD8_TRSPZ014171_u673_1_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ014460_u673_1-006',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014460_u673_1_40X-006",
    #     'tiff': "225_panCK CD8_TRSPZ014460_u673_1_40X-006.tif"
    # },
    # {
    #     'name': 'TRSPZ014459_u673_1',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014459_u673_1_40X",
    #     'tiff': "225_panCK CD8_TRSPZ014459_u673_1_40X.tif"
    # },
    # {
    #     'name': 'TRSPZ014174_u673_1-001',
    #     'folder': r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014174_u673_1_40X-001",
    #     'tiff': "225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif"
    # }
]

for sample in samples:
    sample['resolution_microns'] = 0.25


def main():
    for sample in samples:
        try:
            folder_path = sample['folder']
            cd8_path = os.path.join(folder_path, "cd8.hdf5")
            krt_path = os.path.join(folder_path, "krt.hdf5")
            ref_tiff = os.path.join(folder_path, sample['tiff'])
            output_dir = os.path.join(folder_path, "output_3")
            resulotion = sample['resolution_microns']
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, "gini_results.csv")

            print(f"\n{'=' * 80}")
            print(f"Processing: {sample['name']}")
            print(f"{'=' * 80}")

            # Load mask and identify tissue regions
            print("Loading KRT mask...")
            mask, krt_label, tissue_label = load_krt_mask(krt_path)

            # Extract all CD8 coordinates
            print("Extracting CD8 coordinates...")
            all_coords = extract_label2_coords(cd8_path)
            print(f"Total CD8 points: {len(all_coords)}")

            # Separate coordinates by tissue type
            print("Separating coordinates by tissue type...")
            krt_coords, nonkrt_coords = separate_coords_by_tissue(all_coords, mask, krt_label, tissue_label)
            print(f"KRT points: {len(krt_coords)}, Non-KRT points: {len(nonkrt_coords)}")

            import csv

            with open(results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Bandwidth", "Tissue_Type", "Gini", "N_Points"])

                # for bw in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
                for bw in [5, 10, 25, 50, 100]:
                    print(f"\n{'=' * 60}")
                    print(f"Processing bandwidth: {bw} microns")
                    print(f"{'=' * 60}")

                    results = estimate_kde(krt_coords, nonkrt_coords, resulotion,
                                           output_dir, ref_tiff,
                                           bandwidth=bw, tile_level=2, grid_resolution=1000, map_size=3000
                                           )

                    # Write results for both tissue types
                    if results['krt']['gini'] is not None:
                        writer.writerow([bw, "krt", f"{results['krt']['gini']:.6f}", results['krt']['n_points']])
                    if results['nonkrt']['gini'] is not None:
                        writer.writerow([bw, "nonkrt", f"{results['nonkrt']['gini']:.6f}", results['nonkrt']['n_points']])

                    f.flush()  # Write to disk immediately

            print(f"\n{'=' * 60}")
            print(f"All results saved to: {results_file}")
            print(f"{'=' * 60}")

            import matplotlib.pyplot as plt

            # Read the CSV file
            import pandas as pd
            df = pd.read_csv(results_file)

            # Separate by tissue type
            krt_data = df[df['Tissue_Type'] == 'krt']
            nonkrt_data = df[df['Tissue_Type'] == 'nonkrt']

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(krt_data['Bandwidth'], krt_data['Gini'], 'o-', label='KRT', linewidth=2, markersize=8)
            plt.plot(nonkrt_data['Bandwidth'], nonkrt_data['Gini'], 's-', label='Non-KRT', linewidth=2, markersize=8)

            plt.xlabel('Bandwidth in microns', fontsize=12)
            plt.ylabel('Gini Coefficient', fontsize=12)
            plt.title('Bandwidth vs Gini Coefficient', fontsize=14)
            # plt.xscale('log')  # Log scale for bandwidth since values span multiple orders of magnitude
            # plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.xticks(krt_data['Bandwidth'])
            plt.tight_layout()

            # Save the plot
            plot_file = os.path.join(output_dir, "bandwidth_vs_gini.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_file}")
            plt.close()
        except Exception as e:
            print(e)



if __name__ == '__main__':
    main()
