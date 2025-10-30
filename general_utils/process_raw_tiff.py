import tifffile
from osgeo import gdal
import tempfile
import os

tiff_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\225_panCK CD8_TRSPZ005647_u673_1_40X.tif"
output_tiff = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\225_panCK CD8_TRSPZ005647_u673_1_40X_w_overviews.tif"

# Load level 2 with tifffile
with tifffile.TiffFile(tiff_path) as tif:
    image = tif.pages[2].asarray()
    print(f"Loaded level 2 - Shape: {image.shape}")

# Save temporarily uncompressed
temp_tiff = tempfile.mktemp(suffix='.tif')
tifffile.imwrite(temp_tiff, image, photometric='rgb')

# Convert to JPEG-compressed TIFF with GDAL
gdal.Translate(
    output_tiff,
    temp_tiff,
    creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR']
)
os.remove(temp_tiff)
print(f"Saved as compressed TIFF: {output_tiff}")

# Add overviews
ds = gdal.Open(output_tiff, gdal.GA_Update)
ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
ds = None
print("Overviews added - Done!")