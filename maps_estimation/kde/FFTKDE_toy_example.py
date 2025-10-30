import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import pandas as pd
folder_dir = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X"
csv_path = fr"{folder_dir}\detections_from_iris.csv"

df = pd.read_csv(csv_path)
if 'category' in df.columns:
    df = df[df['category'] == 'Category 2']
x, y = df['x'].values, df['y'].values
print(f"Loaded {len(x)} points")
data = np.column_stack([x, y])
# Create 2D data in the range [0, 100]
np.random.seed(42)
# data = np.vstack([
#     np.random.randn(300, 2) * 5 + [30, 40],   # cluster 1 around (30, 40)
#     np.random.randn(200, 2) * 8 + [70, 60]    # cluster 2 around (70, 60)
# ])

# Clip to ensure points stay in [0, 100] range
# data = np.clip(data, 0, 100)

# Fit and evaluate with number of grid points
kde = FFTKDE(kernel='gaussian', bw=1)
kde.fit(data)

# Pass integer for square grid, or tuple for different dimensions
grid, points = kde.evaluate(int(80000 / 60))  # 50x50 grid

# Extract x and y coordinates
x_grid = np.unique(grid[:, 0])
y_grid = np.unique(grid[:, 1])
density = points.reshape(len(y_grid), len(x_grid)).T

print(f"Grid shape: {grid.shape}")
print(f"Points shape: {points.shape}")
print(f"x_grid shape: {x_grid.shape}")
print(f"y_grid shape: {y_grid.shape}")
print(f"Density shape: {density.shape}")

density = (density - density.min()) / (density.max() - density.min())

# Plot
plt.figure(figsize=(10, 5), dpi=300)  # Increase DPI from default 100

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
plt.title('Data')

plt.subplot(1, 2, 2)
plt.imshow(np.log10(density + 1e-10), extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
           origin='lower', cmap='hot')
# plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=5, c='cyan')
plt.colorbar(label='Density')
plt.title('FFTKDE')

plt.tight_layout()
plt.show()