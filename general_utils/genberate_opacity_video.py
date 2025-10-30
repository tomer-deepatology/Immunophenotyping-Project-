import cv2
import numpy as np
import tifffile
from PIL import Image

# ===== PARAMETERS =====
tile_path = r"C:\Users\tomer\Desktop\data\demo_sunday\demo.tif"
heatmap_path = r"C:\Users\tomer\Desktop\data\demo_sunday\bw_None_heatmap.tif"
output_video = r"C:\Users\tomer\Desktop\data\demo_sunday\overlay_video.mp4"
fps = 3  # Frames per second
num_frames = 50  # Total frames in video
min_opacity = 0.0  # Starting opacity (0 = invisible)
max_opacity = 1.0  # Ending opacity (1 = fully visible)
# ======================

# Load images
tile_img = tifffile.imread(tile_path)
heatmap_img = tifffile.imread(heatmap_path)

# Ensure same dimensions
assert tile_img.shape[:2] == heatmap_img.shape[:2], "Images must have same dimensions!"

height, width = tile_img.shape[:2]

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Generate frames with varying opacity
for i in range(num_frames):
    # Calculate opacity for this frame (linear interpolation)
    alpha = min_opacity + (max_opacity - min_opacity) * (i / (num_frames - 1))

    # Blend images: result = tile * (1 - alpha) + heatmap * alpha
    blended = cv2.addWeighted(tile_img, 1 - alpha, heatmap_img, alpha, 0)

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2BGR)

    video_writer.write(frame_bgr)

    if (i + 1) % 10 == 0:
        print(f"Frame {i + 1}/{num_frames} (opacity: {alpha:.2f})")

video_writer.release()
print(f"\nVideo saved to: {output_video}")