import numpy as np
from PIL import Image


def generate_red_points(output_file='red_points.png', width=512, height=512, k=3, n_points=1000):
    # Create image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Random Gaussian centers
    centers = np.random.rand(k, 2) * [width, height]
    sigmas = np.random.rand(k, 2) * 30 + 20

    # Generate points from Gaussian mixture
    cluster = np.random.choice(k, n_points)
    points = np.random.randn(n_points, 2) * sigmas[cluster] + centers[cluster]
    intensities = np.random.randint(100, 256, n_points)

    # Draw points
    for i in range(n_points):
        x, y = int(points[i, 0]), int(points[i, 1])
        if 0 <= x < width and 0 <= y < height:
            img[y, x, 0] = intensities[i]

    Image.fromarray(img).save(output_file)


# Usage
generate_red_points('red_points.png', width=512, height=512, k=20, n_points=1000)