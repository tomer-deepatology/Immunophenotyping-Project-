import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os


def estimate_kde_from_chunk(chunk_csv_path, output_path, bandwidth):
    # Load CSV
    df = pd.read_csv(chunk_csv_path)

    # Get local coordinates
    x = df['x_local'].values
    y = df['y_local'].values

    # Create KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bandwidth)

    # Create grid for evaluation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
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

    # Plot
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

    plt.savefig(output_path, dpi=1200, format='jpeg', pil_kwargs={'optimize': True})
    plt.close()


def main():
    chunk_csv = r"C:\Users\User\Desktop\data\sample 2\jpeg version\2025-09-29_full_detections_fitlered.csv"
    for bandwidth in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        output_path = fr'C:\Users\User\Desktop\data\sample 2\jpeg version\kde\kde_chunk_bw_{bandwidth}.png'
        estimate_kde_from_chunk(chunk_csv, output_path=output_path, bandwidth=bandwidth)


if __name__ == '__main__':
    main()
