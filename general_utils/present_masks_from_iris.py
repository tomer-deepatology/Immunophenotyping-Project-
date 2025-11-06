import h5py
import numpy as np
import json
from PIL import Image


def present_masks_from_iris(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Load the mask
        mask = f['wsi_masks/predicted_region_mask_l0'][:]

        # Load the presentation metadata
        presentation_data = json.loads(f['wsi_presentation/masks'][()])

        # Extract label-to-color mapping
        labels = presentation_data[0]['data']

        # Create RGB image
        height, width = mask.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for label_info in labels:
            label_value = label_info['label']
            color_hex = label_info['color']
            name = label_info['textgui']

            # Convert hex color to RGB
            color_hex = color_hex.lstrip('#')
            r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)

            # Apply color to mask
            rgb_image[mask == label_value] = [r, g, b]

            print(f"Label {label_value} ({name}): RGB({r}, {g}, {b})")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10), dpi=100)  # medium DPI and size
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title("Segmentation Mask (Preview)", fontsize=14)
        plt.show()

        print(f"Image saved as 'segmentation_mask.jpg' with size {width}x{height}")


def main():
    # file_path = "/mnt/c/Users/tomer/Desktop/data/project 1/225_panCK CD8_TRSPZ005647_u673_1_40X/krt.hdf5"

    file_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\krt.hdf5"
    present_masks_from_iris(file_path)


if __name__ == '__main__':
    main()
