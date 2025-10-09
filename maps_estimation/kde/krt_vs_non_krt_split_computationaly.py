import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
from osgeo import gdal
import tifffile
import tempfile
import os
import json
from shapely.geometry import Point, Polygon
import cv2

Image.MAX_IMAGE_PIXELS = None


def compute_and_save_kde(csv_path, annotations_path, output_base, bandwidth, reference_image_path):
    """Compute KDE and save raw density arrays"""
    # Load reference image
    with tifffile.TiffFile(reference_image_path) as tif:
        ref_image = tif.pages[0].asarray()
        ref_height, ref_width = ref_image.shape[:2]

    # Load points and polygons
    df = pd.read_csv(csv_path)
    with open(annotations_path, 'r') as f:
        geojson = json.load(f)

    # Get bounding box with margin
    all_coords = np.array([coord for feature in geojson['features']
                           for coord in feature['geometry']['coordinates'][0]])
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)

    margin = 100
    x_min = max(0, int(x_min - margin))
    x_max = min(ref_width, int(x_max + margin))
    y_min = max(0, int(y_min - margin))
    y_max = min(ref_height, int(y_max + margin))

    # Filter points to bbox and separate KRT/non-KRT
    df = df[(df['x'] >= x_min) & (df['x'] <= x_max) &
            (df['y'] >= y_min) & (df['y'] <= y_max)].copy()

    polygons = [Polygon(f['geometry']['coordinates'][0]).buffer(50) for f in geojson['features']]
    df['inside_krt'] = df.apply(lambda r: any(p.contains(Point(r['x'], r['y'])) for p in polygons), axis=1)

    df_krt = df[df['inside_krt']]
    df_non_krt = df[~df['inside_krt']]
    print(f"KRT: {len(df_krt)}, Non-KRT: {len(df_non_krt)}")

    # Create KDE grid for cropped region only
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, crop_width),
                         np.linspace(y_min, y_max, crop_height))

    # Evaluate KDEs
    print("Computing KRT KDE...")
    kde_krt = gaussian_kde(np.vstack([df_krt['x'].values, df_krt['y'].values]), bw_method=bandwidth)
    density_krt = kde_krt(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    print("Computing non-KRT KDE...")
    kde_non_krt = gaussian_kde(np.vstack([df_non_krt['x'].values, df_non_krt['y'].values]), bw_method=bandwidth)
    density_non_krt = kde_non_krt(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Crop and save reference image
    ref_cropped = ref_image[y_min:y_max, x_min:x_max]

    # Save raw KDE arrays and cropped image
    kde_file = f'{output_base}_raw_kde.npz'
    np.savez_compressed(kde_file,
                        density_krt=density_krt,
                        density_non_krt=density_non_krt,
                        ref_cropped=ref_cropped,
                        x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max)
    print(f"Saved raw KDE arrays and cropped reference image to: {kde_file}")


def create_visualizations_from_kde(kde_file, output_base):
    """Load KDE arrays and create visualizations"""
    # Load KDE arrays and cropped reference image
    print(f"Loading KDE from {kde_file}")
    data = np.load(kde_file)
    density_krt = data['density_krt']
    density_non_krt = data['density_non_krt']
    ref_cropped = data['ref_cropped']
    x_min, x_max = int(data['x_min']), int(data['x_max'])
    y_min, y_max = int(data['y_min']), int(data['y_max'])

    # Normalize
    krt_norm = (density_krt - density_krt.min()) / (density_krt.max() - density_krt.min())
    non_krt_norm = (density_non_krt - density_non_krt.min()) / (density_non_krt.max() - density_non_krt.min())

    # Create heatmaps with black for zero values
    krt_colored = plt.get_cmap('hot')(krt_norm)[:, :, :3]
    krt_colored[krt_norm <= np.quantile(krt_norm, 0.1)] = [0, 0, 0]
    krt_img = Image.fromarray((krt_colored * 255).astype(np.uint8))

    non_krt_colored = plt.get_cmap('gist_earth')(non_krt_norm)[:, :, :3]
    non_krt_colored[non_krt_norm <= np.quantile(non_krt_norm, 0.1)] = [0, 0, 0]
    non_krt_img = Image.fromarray((non_krt_colored * 255).astype(np.uint8))

    krt_density_norm = (density_krt - density_krt.min()) / (density_krt.max() - density_krt.min())
    krt_weight = (krt_density_norm ** 0.1)[:, :, np.newaxis]  # Power < 1 emphasizes red more

    # Blend: full non-KRT where density_krt is low, full KRT where density_krt is high
    combined_colored = krt_weight * krt_colored + (1 - krt_weight) * non_krt_colored
    combined_heatmap = Image.fromarray((combined_colored * 255).astype(np.uint8))

    # Create overlays using loaded cropped reference
    ref_cropped_pil = Image.fromarray(ref_cropped)
    krt_overlay = Image.blend(ref_cropped_pil, krt_img, alpha=0.7)
    non_krt_overlay = Image.blend(ref_cropped_pil, non_krt_img, alpha=0.7)
    combined_overlay = Image.blend(ref_cropped_pil, combined_heatmap, alpha=0.7)

    # Save all as TIFF
    for name, img in [('krt_heatmap', krt_img), ('non_krt_heatmap', non_krt_img),
                      ('combined_heatmap', combined_heatmap), ('krt_overlay', krt_overlay),
                      ('non_krt_overlay', non_krt_overlay), ('combined_overlay', combined_overlay)]:
        temp_tiff = tempfile.mktemp(suffix='.tif')
        tifffile.imwrite(temp_tiff, np.array(img), photometric='rgb')
        output_path = f'{output_base}_{name}.tif'
        gdal.Translate(output_path, temp_tiff,
                       creationOptions=['COMPRESS=JPEG', 'JPEG_QUALITY=95', 'TILED=YES', 'PHOTOMETRIC=YCBCR'])
        os.remove(temp_tiff)
        ds = gdal.Open(output_path, gdal.GA_Update)
        ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32])
        ds = None
        print(f"Saved: {output_path}")


def create_alpha_videos(kde_file, output_base, fps=10, num_frames=100):
    """Create videos showing alpha transition from 0.0 to 1.0"""
    print(f"Creating alpha transition videos...")

    # Load KDE arrays and cropped reference image
    data = np.load(kde_file)
    density_krt = data['density_krt']
    density_non_krt = data['density_non_krt']
    ref_cropped = data['ref_cropped']

    # Normalize
    krt_norm = (density_krt - density_krt.min()) / (density_krt.max() - density_krt.min())
    non_krt_norm = (density_non_krt - density_non_krt.min()) / (density_non_krt.max() - density_non_krt.min())

    # Create heatmaps with black for zero values
    krt_colored = plt.get_cmap('hot')(krt_norm)[:, :, :3]
    krt_colored[krt_norm <= np.quantile(krt_norm, 0.1)] = [0, 0, 0]
    krt_img = Image.fromarray((krt_colored * 255).astype(np.uint8))

    non_krt_colored = plt.get_cmap('gist_earth')(non_krt_norm)[:, :, :3]
    non_krt_colored[non_krt_norm <= np.quantile(non_krt_norm, 0.1)] = [0, 0, 0]
    non_krt_img = Image.fromarray((non_krt_colored * 255).astype(np.uint8))

    krt_density_norm = (density_krt - density_krt.min()) / (density_krt.max() - density_krt.min())
    krt_weight = (krt_density_norm ** 0.1)[:, :, np.newaxis]

    combined_colored = krt_weight * krt_colored + (1 - krt_weight) * non_krt_colored
    combined_heatmap = Image.fromarray((combined_colored * 255).astype(np.uint8))

    ref_cropped_pil = Image.fromarray(ref_cropped)

    # Get dimensions (note: cv2 uses width, height order)
    height, width = ref_cropped.shape[:2]

    # Define video codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    videos = {
        'krt': (krt_img, f'{output_base}_krt_alpha_transition.mp4'),
        'non_krt': (non_krt_img, f'{output_base}_non_krt_alpha_transition.mp4'),
        'combined': (combined_heatmap, f'{output_base}_combined_alpha_transition.mp4')
    }

    for name, (heatmap, video_path) in videos.items():
        print(f"Creating {name} video...")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Generate frames
        alphas = np.linspace(0.0, 1.0, num_frames)

        for i, alpha in enumerate(alphas):
            # Blend reference image with heatmap
            blended = Image.blend(ref_cropped_pil, heatmap, alpha=alpha)

            # Convert to numpy array and BGR for cv2
            frame = np.array(blended)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_frames} frames")

        out.release()
        print(f"Saved: {video_path}")

    print("All videos created successfully!")


def main():
    ref_image = r"C:\Users\tomer\Desktop\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X_level_2_with_overviews.tif"
    csv_path = r"C:\Users\tomer\Desktop\data\project 1\sample 2\2025-09-29_full_detections.csv"
    annotations_path = r"C:\Users\tomer\Desktop\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"
    bw = 0.03
    output = fr"C:\Users\tomer\Desktop\data\project 1\sample 2\kde_krt_comparison\kde_bw_{bw}"

    # Step 1: Compute KDE (run once, takes time)
    # compute_and_save_kde(csv_path, annotations_path, output, bw, ref_image)

    # Step 2: Create visualizations (run multiple times with different settings, fast!)
    kde_file = f'{output}_raw_kde.npz'
    # create_visualizations_from_kde(kde_file, output)

    # Step 3: Create alpha transition videos
    create_alpha_videos(kde_file, output, fps=4, num_frames=20)


if __name__ == '__main__':
    main()