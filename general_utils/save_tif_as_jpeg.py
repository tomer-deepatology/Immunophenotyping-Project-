import tifffile
from PIL import Image
import numpy as np
from pathlib import Path


def tiff_tile_to_jpeg(tiff_path, tile_level):
    """
    Load a specific tile/page from a TIFF file and save it as a JPEG.
    Output path will be the same as input path but with .jpeg extension.

    Parameters:
    - tiff_path: Path to the input TIFF file
    - tile_level: The tile/page number to extract (0-indexed)
    """
    # Generate output path with .jpeg extension
    output_jpeg_path = Path(tiff_path).with_suffix('.jpeg')

    with tifffile.TiffFile(tiff_path) as tif:
        print(f"Total pages in TIFF: {len(tif.pages)}")

        # Get the specific tile
        page = tif.pages[tile_level]
        ref_height, ref_width = page.shape[:2]
        print(f"Reference tile {tile_level}: {ref_width}x{ref_height}")

        # Read the tile data
        tile_data = page.asarray()

        # Convert to PIL Image and save as JPEG
        img = Image.fromarray(tile_data)
        img.save(output_jpeg_path, 'JPEG', quality=95)

        print(f"Saved tile {tile_level} to: {output_jpeg_path}")


# Example usage
if __name__ == '__main__':
    tiff_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ014174_u673_1_40X-001.tif"
    tile_level = 3

    tiff_tile_to_jpeg(tiff_path, tile_level)