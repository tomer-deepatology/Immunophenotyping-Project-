import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def estimate_kde(csv_path, output_base, bandwidth, reference_image_path):
    # Get reference image dimensions
    if reference_image_path.lower().endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(reference_image_path) as tif:
            ref_height, ref_width = tif.pages[0].shape[:2]
            ref_image = tif.pages[0].asarray()
    else:
        with Image.open(reference_image_path) as img:
            ref_width, ref_height = img.size
            ref_image = np.array(img)

    print(f"Reference: {ref_width}x{ref_height}")

    # Load points
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        df = df[df['category'] == 'Category 2']
    x, y = df['x'].values, df['y'].values

    # Create KDE
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

    # Evaluate on grid
    xx, yy = np.meshgrid(np.linspace(0, ref_width, 2000),
                         np.linspace(0, ref_height, 2000))
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Calculate metrics
    density_norm = density.ravel() / density.sum()
    density_nonzero = density_norm[density_norm > 0]

    entropy = -np.sum(density_nonzero * np.log(density_nonzero)) / np.log(len(density_nonzero))

    sorted_d = np.sort(density.ravel())
    n = len(sorted_d)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_d)) / (n * sorted_d.sum()) - (n + 1) / n

    # Create heatmap
    density_norm_img = (density - density.min()) / (density.max() - density.min())
    density_rgb = (plt.get_cmap('hot')(density_norm_img)[:, :, :3] * 255).astype(np.uint8)
    density_resized = Image.fromarray(density_rgb).resize((ref_width, ref_height), Image.BILINEAR)

    # Create overlay (blend heatmap with reference image)
    overlay = Image.blend(Image.fromarray(ref_image), density_resized, alpha=0.5)

    # Save heatmap and overlay
    if reference_image_path.lower().endswith(('.tif', '.tiff')):
        # Save heatmap as TIFF
        heatmap_tiff = f'{output_base}_heatmap.tif'
        temp_tiff = tempfile.mktemp(suffix='.tif')
        tifffile.imwrite(temp_tiff, np.array(density_resized), photometric='rgb')
        gdal.Translate(heatmap_tiff, temp_tiff,
                       creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
        os.remove(temp_tiff)
        ds = gdal.Open(heatmap_tiff, gdal.GA_Update)
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None
        print(f"Saved: {heatmap_tiff}")

        # Save overlay as TIFF
        overlay_tiff = f'{output_base}_overlay.tif'
        temp_tiff = tempfile.mktemp(suffix='.tif')
        tifffile.imwrite(temp_tiff, np.array(overlay), photometric='rgb')
        gdal.Translate(overlay_tiff, temp_tiff,
                       creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
        os.remove(temp_tiff)
        ds = gdal.Open(overlay_tiff, gdal.GA_Update)
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None
        print(f"Saved: {overlay_tiff}")
    else:
        # Save as JPEG
        heatmap_jpeg = f'{output_base}_heatmap.jpg'
        density_resized.save(heatmap_jpeg, 'JPEG', quality=95)
        print(f"Saved: {heatmap_jpeg}")

        overlay_jpeg = f'{output_base}_overlay.jpg'
        overlay.save(overlay_jpeg, 'JPEG', quality=95)
        print(f"Saved: {overlay_jpeg}")

    # Save plot
    plt.figure(figsize=(10, 10))
    plt.imshow(density, extent=[0, ref_width, ref_height, 0], cmap='hot', aspect='auto', origin='upper')
    plt.colorbar(label='Density')
    plt.scatter(x, y, c='blue', s=1, alpha=0.3)
    plt.xlabel('X Local')
    plt.ylabel('Y Local')
    plt.title(f'KDE Distribution (BW={bandwidth})')

    textstr = (f'Bandwidth: {bandwidth}\n\n'
               f'Normalized Entropy: {entropy:.4f}\n'
               f'  (1=homogeneous, 0=clustered)\n\n'
               f'Gini Coefficient: {gini:.4f}\n'
               f'  (0=homogeneous, 1=clustered)')

    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             family='monospace')

    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {'entropy': entropy, 'gini': gini}


def main():
    ref_image = r"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X_level_2_with_overviews.tif"
    csv_path = r"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\2025-09-29_full_detections.csv"

    for bw in tqdm([0.1, 0.2, 0.3]):
        output = fr"C:\Users\perez\Desktop\deepatplogy_very_temp\data\project 1\sample 2\kde_with_overlay\sample_2_kde_bw_{bw}"
        results = estimate_kde(csv_path, output, bw, ref_image)
        print(f"BW {bw}: Entropy={results['entropy']:.4f}, Gini={results['gini']:.4f}\n")


if __name__ == '__main__':
    main()