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

Image.MAX_IMAGE_PIXELS = None


def estimate_kde_krt(csv_path, annotations_path, output_base, bandwidth, reference_image_path):
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
    kde_krt = gaussian_kde(np.vstack([df_krt['x'].values, df_krt['y'].values]), bw_method=bandwidth)
    density_krt = kde_krt(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    kde_non_krt = gaussian_kde(np.vstack([df_non_krt['x'].values, df_non_krt['y'].values]), bw_method=bandwidth)
    density_non_krt = kde_non_krt(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Create heatmaps
    krt_norm = (density_krt - density_krt.min()) / (density_krt.max() - density_krt.min())
    non_krt_norm = (density_non_krt - density_non_krt.min()) / (density_non_krt.max() - density_non_krt.min())


    # Create heatmaps with black for zero values
    krt_colored = plt.get_cmap('hot')(krt_norm)[:, :, :3]
    krt_colored[krt_norm <= np.quantile(krt_norm, 0.01)] = [0, 0, 0]  # Set zero values to black
    krt_img = Image.fromarray((krt_colored * 255).astype(np.uint8))

    non_krt_colored = plt.get_cmap('gist_earth')(non_krt_norm)[:, :, :3]
    non_krt_colored[non_krt_norm <= np.quantile(non_krt_norm, 0.01)] = [0, 0, 0]  # Set zero values to black
    non_krt_img = Image.fromarray((non_krt_colored * 255).astype(np.uint8))

    # Combine by adding (clamp to 255)
    # combined_heatmap = Image.fromarray(np.clip(np.array(krt_img) + np.array(non_krt_img), 0, 255).astype(np.uint8))
    combined_heatmap = Image.blend(krt_img, non_krt_img, alpha=0.5)

    # Crop reference image and create overlays
    ref_cropped = Image.fromarray(ref_image[y_min:y_max, x_min:x_max])
    krt_overlay = Image.blend(ref_cropped, krt_img, alpha=0.3)
    non_krt_overlay = Image.blend(ref_cropped, non_krt_img, alpha=0.3)
    combined_overlay = Image.blend(ref_cropped, combined_heatmap, alpha=0.5)

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


def main():
    ref_image = r"C:\Users\tomer\Desktop\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X_level_2_with_overviews.tif"
    csv_path = r"C:\Users\tomer\Desktop\data\project 1\sample 2\2025-09-29_full_detections.csv"
    annotations_path = r"C:\Users\tomer\Desktop\data\project 1\sample 2\225_panCK CD8_TRSPZ012209_u673_2_40X.tif - Series 0.geojson"
    bw = 0.05
    output = fr"C:\Users\tomer\Desktop\data\project 1\sample 2\kde_krt_comparison\kde_bw_{bw}"
    estimate_kde_krt(csv_path, annotations_path, output, bw, ref_image)


if __name__ == '__main__':
    main()