import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.cm as cm

# Load data and image
# csv_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.csv"
# img_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations\chunk_183_y16384_x14336\chunk_183_y16384_x14336.png"


csv_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations_4096\chunk_47_y16384_x12288\chunk_47_y16384_x12288.csv"
img_path = r"C:\Users\User\Desktop\data\sample 2\output_chunks_with_annotations_4096\chunk_47_y16384_x12288\chunk_47_y16384_x12288.png"

df = pd.read_csv(csv_path)
base_img = Image.open(img_path).convert('RGBA').resize((800, 800))  # Back to 800x800
x, y = df['x_local'].values, df['y_local'].values
xy = np.vstack([x, y])

w, h = base_img.size
xx, yy = np.meshgrid(np.linspace(0, df['x_local'].max(), 100), np.linspace(0, df['y_local'].max(), 100))
positions = np.vstack([xx.ravel(), yy.ravel()])

update_pending = None


def update_display():
    global update_pending
    update_pending = None

    bw = bw_slider.get()
    alpha = alpha_slider.get()

    kde = gaussian_kde(xy, bw_method=bw)
    density = kde(positions).reshape(xx.shape)
    density = (density - density.min()) / (density.max() - density.min())

    heatmap = (cm.hot(density) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize((w, h)).convert('RGBA')

    result = Image.blend(base_img, heatmap_img, alpha=alpha)
    photo = ImageTk.PhotoImage(result)
    label.config(image=photo)
    label.image = photo
    bw_label.config(text=f'Bandwidth: {bw:.3f}')
    alpha_label.config(text=f'KDE Opacity: {alpha:.2f}')


def schedule_update(v):
    global update_pending
    if update_pending:
        root.after_cancel(update_pending)
    update_pending = root.after(100, update_display)


root = tk.Tk()
root.title("KDE Bandwidth Adjuster")

label = tk.Label(root)
label.pack(pady=10)

bw_label = tk.Label(root, text='Bandwidth: 0.100', font=('Arial', 12))
bw_label.pack(pady=5)

bw_slider = tk.Scale(root, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                     length=600, width=20, command=schedule_update)
bw_slider.set(0.1)
bw_slider.pack(pady=5)

alpha_label = tk.Label(root, text='KDE Opacity: 0.50', font=('Arial', 12))
alpha_label.pack(pady=5)

alpha_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                        length=600, width=20, command=schedule_update)
alpha_slider.set(0.5)
alpha_slider.pack(pady=5)

update_display()
root.mainloop()