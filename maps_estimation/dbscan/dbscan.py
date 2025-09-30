import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw

# Load data and image
csv_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.csv"
img_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.png"

df = pd.read_csv(csv_path)
img = Image.open(img_path)

# Run DBSCAN
X = df[['x_local', 'y_local']].values
clustering = DBSCAN(eps=150, min_samples=5).fit(X)
df['cluster'] = clustering.labels_

# Draw clusters
draw = ImageDraw.Draw(img)
colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

for i, row in df.iterrows():
    x, y = row['x_local'], row['y_local']
    cluster = row['cluster']
    color = 'gray' if cluster == -1 else colors[cluster % len(colors)]
    draw.ellipse([x-3, y-3, x+3, y+3], fill=color)

img.show()
print(f"Found {len(set(clustering.labels_)) - 1} clusters")
print(f"Noise points: {sum(clustering.labels_ == -1)}")