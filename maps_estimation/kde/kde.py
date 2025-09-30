import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os


def estimate_kde_from_chunk(chunk_csv_path, output_path):
    # Load CSV
    df = pd.read_csv(chunk_csv_path)

    # Get local coordinates
    x = df['x_local'].values
    y = df['y_local'].values

    # Create KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.15)

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

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(density, extent=[x_min, x_max, y_max, y_min],
               cmap='hot', aspect='auto', origin='upper')
    plt.colorbar(label='Density')
    plt.scatter(x, y, c='blue', s=1, alpha=0.3)
    plt.xlabel('X Local')
    plt.ylabel('Y Local')
    plt.title('KDE Distribution of Points')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"KDE plot saved to {output_path}")
    return kde


# Usage - example for one chunk
chunk_csv = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.csv"
estimate_kde_from_chunk(chunk_csv, output_path='kde_chunk_0.15.png')