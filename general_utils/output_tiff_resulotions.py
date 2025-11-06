import tifffile

tiff_path = "/mnt/c/Users/tomer/Desktop/data/project 1/225_panCK CD8_TRSPZ005647_u673_1_40X/225_panCK CD8_TRSPZ005647_u673_1_40X.tif"

with tifffile.TiffFile(tiff_path) as tif:
    print(f"Total pages: {len(tif.pages)}")
    for i, page in enumerate(tif.pages):
        print(f"Level {i}: {page.shape[1]}x{page.shape[0]}")