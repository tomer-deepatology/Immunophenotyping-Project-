import tifffile
from PIL import Image, ImageDraw
import pandas as pd
import os


def split_and_mark_chunks(tiff_path, csv_path, output_path, chunk_size, level=2, jpeg_quality=95):
    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df[df['category'] == 'Category 2']

    with tifffile.TiffFile(tiff_path) as tif:
        # Load specified level
        image = tif.pages[level].asarray()
        print(f"Loaded level {level} - Shape: {image.shape}")

        h, w = image.shape[:2]

        # Save chunks with points marked
        chunk_num = 0
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                # Find points within this chunk
                points_in_chunk = df[
                    (df['x'] >= x) & (df['x'] < x + chunk_size) &
                    (df['y'] >= y) & (df['y'] < y + chunk_size)
                ].copy()

                # Skip if no points
                if len(points_in_chunk) == 0:
                    chunk_num += 1
                    continue

                # Create folder for this chunk
                chunk_name = f'chunk_{chunk_num}_y{y}_x{x}'
                chunk_folder = os.path.join(output_path, chunk_name)
                os.makedirs(chunk_folder, exist_ok=True)

                # Extract chunk
                chunk = image[y:y + chunk_size, x:x + chunk_size].copy()

                # Add local coordinates
                points_in_chunk['x_local'] = points_in_chunk['x'] - x
                points_in_chunk['y_local'] = points_in_chunk['y'] - y

                # Draw points on chunk
                img = Image.fromarray(chunk)
                draw = ImageDraw.Draw(img)

                for _, point in points_in_chunk.iterrows():
                    px, py = point['x_local'], point['y_local']
                    draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill='red', outline='yellow')

                # Save image as JPEG with compression and CSV
                img.save(f'{chunk_folder}/{chunk_name}.jpg', 'JPEG', quality=jpeg_quality)
                points_in_chunk.to_csv(f'{chunk_folder}/{chunk_name}.csv', index=False)

                print(f"Saved chunk {chunk_num} with {len(points_in_chunk)} points")
                chunk_num += 1


def main():
    tiff_path = r"C:\Users\User\Desktop\data\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X.tif"
    csv_path = r"C:\Users\User\Desktop\data\sample 2\2025-09-29_full_detections.csv"
    chunk_size = 100000
    output_folder = fr"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations_{chunk_size}"

    split_and_mark_chunks(
        tiff_path=tiff_path,
        csv_path=csv_path,
        output_path=output_folder,
        chunk_size=chunk_size,
        jpeg_quality=95  # Optional: adjust quality (1-100, default 95)
    )


if __name__ == '__main__':
    main()