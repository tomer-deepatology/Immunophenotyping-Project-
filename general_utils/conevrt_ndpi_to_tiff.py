import openslide
import tifffile
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import tempfile
import os

# ===== PARAMETERS =====
file_path = r"C:\Users\tomer\Desktop\data\demo_sunday\OS-2.ndpi"
output_path = r"C:\Users\tomer\Desktop\data\demo_sunday\demo_best_resolution.tif"
level = 0  # Which pyramid level (0 = highest resolution)
tile_size = 4096  # Size of chunks to read at a time
jpeg_quality = 90  # JPEG compression quality
overview_levels = [2, 4, 8, 16, 32]  # Overview downsampling factors
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

# Create temporary uncompressed TIFF
temp_tiff = tempfile.mktemp(suffix='.tif')
print(f"Creating temporary file: {temp_tiff}")

# Create output array in chunks using memmap (disk-backed array)
output_array = np.memmap(temp_tiff.replace('.tif', '_temp.dat'),
                         dtype='uint8', mode='w+',
                         shape=(height, width, 3))

# Read and write in tiles with progress bar
print("Reading tiles from NDPI...")
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

slide.close()

print("Saving as uncompressed TIFF...")
tifffile.imwrite(temp_tiff, output_array, photometric='rgb')

# Clean up memmap
del output_array
os.remove(temp_tiff.replace('.tif', '_temp.dat'))

# Compress with GDAL
print(f"Compressing with JPEG (quality={jpeg_quality})...")
gdal.Translate(
    output_path,
    temp_tiff,
    creationOptions=[
        'COMPRESS=JPEG',
        f'JPEG_QUALITY={jpeg_quality}',
        'TILED=YES',
        'PHOTOMETRIC=YCBCR'
    ]
)

# Remove temporary file
os.remove(temp_tiff)

# Add overviews
print(f"Building overviews: {overview_levels}...")
ds = gdal.Open(output_path, gdal.GA_Update)
ds.BuildOverviews('AVERAGE', overview_levels)
ds = None

print(f"Image saved with overviews to: {output_path}")
print("Done!")