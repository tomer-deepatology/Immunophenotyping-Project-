import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ===== PARAMETERS =====
csv_path = r"C:\Users\tomer\Desktop\data\demo_sunday\report\2025-10-30_full_detections.csv"
output_txt = r"C:\Users\tomer\Desktop\data\demo_sunday\knn_statistics.txt"
output_csv = r"C:\Users\tomer\Desktop\data\demo_sunday\knn_per_point.csv"
category_filter = 'Category 2'
k = 50
pixel_to_um = 0.2271
# ======================

# Load and filter
df = pd.read_csv(csv_path)
df = df[df['category'] == category_filter]
points = df[['x', 'y']].values
print(f"Points: {len(points)}")

# KNN
knn = NearestNeighbors(n_neighbors=k+1)
knn.fit(points)
distances, indices = knn.kneighbors(points)
distances = distances[:, 1:] * pixel_to_um  # Remove self, convert to µm

# Per-point stats
per_point_stats = pd.DataFrame({
    'mean_distance_um': distances.mean(axis=1),
    'median_distance_um': np.median(distances, axis=1),
    'min_distance_um': distances.min(axis=1),
    'max_distance_um': distances.max(axis=1),
    'std_distance_um': distances.std(axis=1)
})
per_point_stats.to_csv(output_csv, index=False)

# Aggregated stats
all_distances = distances.flatten()
stats_text = f"""{"="*50}
KNN DISTANCE STATISTICS (K={k})
{"="*50}
Category: {category_filter}
Number of points: {len(points)}
Resolution: {pixel_to_um} µm/pixel

AGGREGATED METRICS:
Average distance (all): {all_distances.mean():.2f} µm
Median distance (all):  {np.median(all_distances):.2f} µm

PER-POINT LOCAL ENVIRONMENT:
Mean of local means:    {per_point_stats['mean_distance_um'].mean():.2f} µm
Median of local means:  {per_point_stats['mean_distance_um'].median():.2f} µm
Mean nearest neighbor:  {per_point_stats['min_distance_um'].mean():.2f} µm
Median nearest neighbor:{per_point_stats['min_distance_um'].median():.2f} µm
{"="*50}
"""

print(stats_text)
with open(output_txt, 'w') as f:
    f.write(stats_text)

print(f"\nSaved:\n- {output_txt}\n- {output_csv}")