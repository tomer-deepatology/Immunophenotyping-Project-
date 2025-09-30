import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
import matplotlib.cm as cm
import tifffile
import os

Image.MAX_IMAGE_PIXELS = None

def estimate_kde_from_chunk(chunk_csv_path, output_path_base, bandwidth, reference_image_path):
    # Load reference image to get dimensions
    if reference_image_path.lower().endswith('.tif') or reference_image_path.lower().endswith('.tiff'):
        with tifffile.TiffFile(reference_image_path) as tif:
            ref_shape = tif.pages[0].shape
    else:  # JPEG or other formats
        with Image.open(reference_image_path) as ref_img:
            ref_width, ref_height = ref_img.size  # Only reads metadata, not pixel data
            ref_shape = (ref_height, ref_width)

    ref_height, ref_width = ref_shape[:2]
    print(f"Reference image size: {ref_width} x {ref_height}")

    # Load CSV
    df = pd.read_csv(chunk_csv_path)

    # Get local coordinates
    x = df['x_local'].values
    y = df['y_local'].values

    # Create KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bandwidth)

    # Create grid for evaluation - USE REFERENCE IMAGE DIMENSIONS
    xx, yy = np.meshgrid(
        np.linspace(0, ref_width, ref_width),  # From 0 to image width
        np.linspace(0, ref_height, ref_height)  # From 0 to image height
    )

    # Evaluate KDE on grid
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    # Calculate sparsity metrics
    density_flat = density.ravel()
    density_norm = density_flat / density_flat.sum()
    density_norm_nonzero = density_norm[density_norm > 0]

    # Normalized Entropy (0 to 1, higher = more sparse/uniform)
    H = -np.sum(density_norm_nonzero * np.log(density_norm_nonzero))
    H_max = np.log(len(density_norm_nonzero))
    entropy_normalized = H / H_max

    # Gini Coefficient (0 to 1, higher = more clustered)
    sorted_density = np.sort(density_flat)
    n = len(density_flat)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_density)) / (n * np.sum(density_flat)) - (n + 1) / n

    # Save heatmap as JPEG with 'hot' colormap (black->red->yellow->white)
    heatmap_path = f'{output_path_base}_heatmap.jpg'
    # Normalize density to 0-1 range
    density_normalized = (density - density.min()) / (density.max() - density.min())
    # Apply 'hot' colormap
    hot_colormap = cm.get_cmap('hot')
    density_colored = hot_colormap(density_normalized)
    # Convert to RGB (0-255)
    density_rgb = (density_colored[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(density_rgb).save(heatmap_path, 'JPEG', quality=95)
    print(f"Saved heatmap: {ref_width}x{ref_height} - {heatmap_path}")

    # Plot with original point coordinates
    x_min, x_max = 0, ref_width
    y_min, y_max = 0, ref_height

    plt.figure(figsize=(10, 10))
    plt.imshow(density, extent=[x_min, x_max, y_max, y_min],
               cmap='hot', aspect='auto', origin='upper')
    plt.colorbar(label='Density')
    plt.scatter(x, y, c='blue', s=1, alpha=0.3)
    plt.xlabel('X Local')
    plt.ylabel('Y Local')
    plt.title(f'KDE Distribution of Points (BW={bandwidth})')

    # Add text box with metrics and explanation
    textstr = f'Bandwidth: {bandwidth}\n\n'
    textstr += f'Normalized Entropy: {entropy_normalized:.4f}\n'
    textstr += f'  (1 = very homogeneous, 0 = clustered)\n\n'
    textstr += f'Gini Coefficient: {gini:.4f}\n'
    textstr += f'  (0 = very homogeneous, 1 = clustered)'

    # Place text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props,
             family='monospace')

    plt.savefig(f'{output_path_base}.png', dpi=1200, format='png', bbox_inches='tight')
    plt.close()


def main():
    chunk_csv = r"C:\Users\User\Desktop\data\sample 2\jpeg version\2025-09-29_full_detections_fitlered.csv"
    reference_image = r"C:\Users\User\Desktop\data\sample 2\jpeg version\sample_2.jpg"  # Path to your reference image

    for bandwidth in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        output_path_base = fr'C:\Users\User\Desktop\data\sample 2\jpeg version\kde\kde_bandwidth_{bandwidth}'
        estimate_kde_from_chunk(chunk_csv, output_path_base=output_path_base,
                                bandwidth=bandwidth, reference_image_path=reference_image)


if __name__ == '__main__':
    main()