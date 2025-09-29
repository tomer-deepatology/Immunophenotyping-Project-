import tifffile
from PIL import Image
import os

tiff_path = r"C:\Users\User\Desktop\data\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X.tif"

output_folder = r"C:\Users\User\Desktop\data\sample 2\output_chunks"
os.makedirs(output_folder, exist_ok=True)

# with tifffile.TiffFile(tiff_path) as tif:
#     # Load level 2
#     image = tif.pages[2].asarray()
#     print(f"Loaded level 2 - Shape: {image.shape}")
#
#     # Define chunk size
#     chunk_size = 2048  # Adjust as needed
#     h, w = image.shape[:2]
#
#     # Save chunks
#     chunk_num = 0
#     for y in range(0, h, chunk_size):
#         for x in range(0, w, chunk_size):
#             chunk = image[y:y + chunk_size, x:x + chunk_size]
#             img = Image.fromarray(chunk)
#             img.save(f'{output_folder}/chunk_{chunk_num}_y{y}_x{x}.png')
#             chunk_num += 1
#             print(f"Saved chunk {chunk_num}")
#
#     print(f"Total chunks saved: {chunk_num}")


with tifffile.TiffFile(tiff_path) as tif:
    for i, page in enumerate(tif.pages):
        print(f"Shape: {page.shape}")