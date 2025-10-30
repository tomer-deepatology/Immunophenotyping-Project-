import openslide
import tifffile
import numpy as np
from tqdm import tqdm

# ===== PARAMETERS =====
file_path = r"C:\Users\tomer\Desktop\data\demo_sunday\OS-2.ndpi"
output_path = r"C:\Users\tomer\Desktop\data\demo_sunday\demo_best_resolution.tif"
level = 0  # Which pyramid level (0 = highest resolution)
tile_size = 4096  # Size of chunks to read at a time
# ======================

slide = openslide.OpenSlide(file_path)

# Get dimensions for this level
width, height = slide.level_dimensions[level]
print(f"Level {level}: {width} x {height}")

# Calculate number of tiles needed
n_tiles_x = int(np.ceil(width / tile_size))
n_tiles_y = int(np.ceil(height / tile_size))
total_tiles = n_tiles_x * n_tiles_y
print(f"Will process {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")

# Create output array in chunks using memmap (disk-backed array)
output_array = np.memmap(output_path.replace('.tif', '_temp.dat'),
                         dtype='uint8', mode='w+',
                         shape=(height, width, 3))

# Read and write in tiles with progress bar
with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            x = tx * tile_size
            y = ty * tile_size

            # Calculate actual tile size (handle edge tiles)
            actual_w = min(tile_size, width - x)
            actual_h = min(tile_size, height - y)

            # Read tile
            tile = slide.read_region((x, y), level, (actual_w, actual_h))
            tile_array = np.array(tile.convert('RGB'))

            # Write to output
            output_array[y:y + actual_h, x:x + actual_w, :] = tile_array

            pbar.update(1)

print("Saving as TIFF...")
tifffile.imwrite(output_path, output_array)

# Clean up
del output_array
import os

os.remove(output_path.replace('.tif', '_temp.dat'))

print(f"Image saved to: {output_path}")
slide.close()