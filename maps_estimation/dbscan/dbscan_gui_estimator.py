import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# Load data and image
csv_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.csv"
img_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.png"

df = pd.read_csv(csv_path)
base_img = Image.open(img_path).convert('RGBA').resize((700, 700))  # Smaller image
X = df[['x_local', 'y_local']].values
colors = [(255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255), (255, 255, 0, 255),
          (0, 255, 255, 255), (255, 0, 255, 255), (255, 128, 0, 255), (128, 0, 255, 255)]

update_pending = None


def update_display():
    global update_pending
    update_pending = None

    eps = eps_slider.get()
    min_samples = min_slider.get()
    opacity = opacity_slider.get()

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = 700 / 2048
    for i, label in enumerate(clustering.labels_):
        x, y = X[i] * scale
        color = (128, 128, 128, int(opacity * 255)) if label == -1 else (*colors[label % len(colors)][:3],
                                                                         int(opacity * 255))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    result = Image.alpha_composite(base_img, overlay)
    photo = ImageTk.PhotoImage(result)
    label_img.config(image=photo)
    label_img.image = photo

    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    info_label.config(
        text=f'Clusters: {n_clusters} | Noise: {sum(clustering.labels_ == -1)} | eps: {eps} | min_samples: {min_samples} | opacity: {opacity:.2f}')


def schedule_update(v):
    global update_pending
    if update_pending:
        root.after_cancel(update_pending)
    update_pending = root.after(100, update_display)


root = tk.Tk()
root.title("DBSCAN Clustering")
root.geometry("800x1000")  # Set window size to fit everything

label_img = tk.Label(root)
label_img.pack(pady=5)

info_label = tk.Label(root, text='', font=('Arial', 10))
info_label.pack(pady=5)

tk.Label(root, text='Epsilon (distance)', font=('Arial', 10)).pack()
eps_slider = tk.Scale(root, from_=10, to=200, resolution=5, orient=tk.HORIZONTAL,
                      length=600, width=20, command=schedule_update)
eps_slider.set(50)
eps_slider.pack(pady=5)

tk.Label(root, text='Min Samples', font=('Arial', 10)).pack()
min_slider = tk.Scale(root, from_=2, to=20, resolution=1, orient=tk.HORIZONTAL,
                      length=600, width=20, command=schedule_update)
min_slider.set(5)
min_slider.pack(pady=5)

tk.Label(root, text='Point Opacity', font=('Arial', 10)).pack()
opacity_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                          length=600, width=20, command=schedule_update)
opacity_slider.set(1.0)
opacity_slider.pack(pady=5)

update_display()
root.mainloop()