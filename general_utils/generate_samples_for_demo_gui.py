import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd


class ClusterPointGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Cluster Point Generator")

        self.width = 512
        self.height = 512
        self.clusters = []
        self.points = []  # Store all points as {'x': x, 'y': y, 'intensity': val}
        self.mode = 'cluster'  # 'cluster', 'add_point', 'delete_point'

        # Setup UI
        self.setup_ui()
        self.create_canvas()

    def setup_ui(self):
        # Control panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Cluster Point Generator", font=('Arial', 14, 'bold')).pack(pady=10)

        # Mode selection
        ttk.Label(control_frame, text="Mode:", font=('Arial', 12, 'bold')).pack(pady=5)
        self.mode_var = tk.StringVar(value='cluster')

        ttk.Radiobutton(control_frame, text="Add Cluster", variable=self.mode_var,
                        value='cluster', command=self.change_mode).pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(control_frame, text="Add Point Manually", variable=self.mode_var,
                        value='add_point', command=self.change_mode).pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(control_frame, text="Delete Point", variable=self.mode_var,
                        value='delete_point', command=self.change_mode).pack(anchor=tk.W, padx=10)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Cluster controls
        ttk.Label(control_frame, text="Cluster Settings:", font=('Arial', 11, 'bold')).pack(pady=5)

        ttk.Label(control_frame, text="Standard Deviation:").pack(pady=2)
        self.std_var = tk.DoubleVar(value=20)
        std_slider = ttk.Scale(control_frame, from_=5, to=100, variable=self.std_var, orient=tk.HORIZONTAL)
        std_slider.pack(fill=tk.X, padx=5)
        self.std_label = ttk.Label(control_frame, text="20.0")
        self.std_label.pack()
        std_slider.config(command=self.update_std_label)

        ttk.Label(control_frame, text="Number of Points:").pack(pady=2)
        self.n_points_var = tk.IntVar(value=100)
        points_slider = ttk.Scale(control_frame, from_=10, to=500, variable=self.n_points_var, orient=tk.HORIZONTAL)
        points_slider.pack(fill=tk.X, padx=5)
        self.points_label = ttk.Label(control_frame, text="100")
        self.points_label.pack()
        points_slider.config(command=self.update_points_label)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Manual point controls
        ttk.Label(control_frame, text="Manual Point Settings:", font=('Arial', 11, 'bold')).pack(pady=5)

        ttk.Label(control_frame, text="Point Intensity:").pack(pady=2)
        self.intensity_var = tk.IntVar(value=200)
        intensity_slider = ttk.Scale(control_frame, from_=100, to=255, variable=self.intensity_var,
                                     orient=tk.HORIZONTAL)
        intensity_slider.pack(fill=tk.X, padx=5)
        self.intensity_label = ttk.Label(control_frame, text="200")
        self.intensity_label.pack()
        intensity_slider.config(command=self.update_intensity_label)

        ttk.Label(control_frame, text="Delete Radius:").pack(pady=2)
        self.delete_radius_var = tk.IntVar(value=5)
        radius_slider = ttk.Scale(control_frame, from_=1, to=20, variable=self.delete_radius_var, orient=tk.HORIZONTAL)
        radius_slider.pack(fill=tk.X, padx=5)
        self.radius_label = ttk.Label(control_frame, text="5")
        self.radius_label.pack()
        radius_slider.config(command=self.update_radius_label)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Buttons
        ttk.Button(control_frame, text="Generate from Clusters", command=self.generate_from_clusters).pack(pady=5,
                                                                                                           fill=tk.X)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Export PNG & CSV", command=self.export_files).pack(pady=10, fill=tk.X)

        # Info
        self.info_label = ttk.Label(control_frame, text="Points: 0 | Clusters: 0", font=('Arial', 10))
        self.info_label.pack(pady=10)

        # Instructions
        self.instructions_label = ttk.Label(control_frame, text="", justify=tk.LEFT, font=('Arial', 9),
                                            foreground='blue')
        self.instructions_label.pack(pady=10)
        self.update_instructions()

    def create_canvas(self):
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=self.width, height=self.height, bg='black', cursor='cross')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        # Create blank image
        self.update_display()

    def update_std_label(self, val):
        self.std_label.config(text=f"{float(val):.1f}")

    def update_points_label(self, val):
        self.points_label.config(text=f"{int(float(val))}")

    def update_intensity_label(self, val):
        self.intensity_label.config(text=f"{int(float(val))}")

    def update_radius_label(self, val):
        self.radius_label.config(text=f"{int(float(val))}")

    def change_mode(self):
        self.mode = self.mode_var.get()
        cursors = {'cluster': 'cross', 'add_point': 'plus', 'delete_point': 'X_cursor'}
        self.canvas.config(cursor=cursors[self.mode])
        self.update_instructions()

    def update_instructions(self):
        instructions = {
            'cluster': "Click to add cluster center\nwith current settings",
            'add_point': "Click to add individual point\nwith current intensity",
            'delete_point': "Click to delete points\nwithin delete radius"
        }
        self.instructions_label.config(text=instructions[self.mode])

    def on_canvas_click(self, event):
        x, y = event.x, event.y

        if self.mode == 'cluster':
            self.add_cluster(x, y)
        elif self.mode == 'add_point':
            self.add_point_manual(x, y)
        elif self.mode == 'delete_point':
            self.delete_points_near(x, y)

    def add_cluster(self, x, y):
        std = self.std_var.get()
        n_points = self.n_points_var.get()

        cluster = {
            'center': (x, y),
            'std': std,
            'n_points': n_points
        }
        self.clusters.append(cluster)

        # Generate points from this cluster
        center = np.array([x, y])
        points = np.random.randn(n_points, 2) * std + center
        intensities = np.random.randint(100, 256, n_points)

        for i in range(n_points):
            px, py = int(points[i, 0]), int(points[i, 1])
            if 0 <= px < self.width and 0 <= py < self.height:
                self.points.append({'x': px, 'y': py, 'intensity': intensities[i]})

        self.update_display()
        self.update_info()

    def add_point_manual(self, x, y):
        intensity = self.intensity_var.get()
        self.points.append({'x': x, 'y': y, 'intensity': intensity})
        self.update_display()
        self.update_info()

    def delete_points_near(self, x, y):
        radius = self.delete_radius_var.get()
        initial_count = len(self.points)

        # Remove points within radius
        self.points = [p for p in self.points
                       if np.sqrt((p['x'] - x) ** 2 + (p['y'] - y) ** 2) > radius]

        deleted = initial_count - len(self.points)
        if deleted > 0:
            self.update_display()
            self.update_info()

    def generate_from_clusters(self):
        if not self.clusters:
            messagebox.showwarning("No Clusters", "Please add at least one cluster first.")
            return

        self.points = []

        for cluster in self.clusters:
            center = np.array(cluster['center'])
            std = cluster['std']
            n_points = cluster['n_points']

            points = np.random.randn(n_points, 2) * std + center
            intensities = np.random.randint(100, 256, n_points)

            for i in range(n_points):
                x, y = int(points[i, 0]), int(points[i, 1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.points.append({'x': x, 'y': y, 'intensity': intensities[i]})

        self.update_display()
        self.update_info()
        messagebox.showinfo("Generated", f"Generated {len(self.points)} points from {len(self.clusters)} clusters!")

    def update_display(self):
        # Create image from points
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for point in self.points:
            x, y = point['x'], point['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                img_array[y, x, 0] = point['intensity']

        # Convert to PhotoImage
        img_pil = Image.fromarray(img_array)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Draw cluster centers if in cluster mode
        if self.mode == 'cluster':
            for i, cluster in enumerate(self.clusters):
                x, y = cluster['center']
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='yellow', outline='white', width=2)
                self.canvas.create_text(x, y - 15, text=f"C{i + 1}", fill='yellow', font=('Arial', 10, 'bold'))

        self.img_array = img_array

    def update_info(self):
        self.info_label.config(text=f"Points: {len(self.points)} | Clusters: {len(self.clusters)}")

    def clear_all(self):
        self.clusters = []
        self.points = []
        self.update_display()
        self.update_info()

    def export_files(self):
        if not self.points:
            messagebox.showwarning("No Points", "Please generate or add points first.")
            return

        # Ask for output file name (without extension)
        file_path = filedialog.asksaveasfilename(
            title="Save files as",
            defaultextension="",
            filetypes=[("All files", "*.*")]
        )

        if not file_path:
            return

        # Remove extension if user added one
        import os
        base_path = os.path.splitext(file_path)[0]

        # Export PNG
        png_path = f"{base_path}.png"
        Image.fromarray(self.img_array).save(png_path)

        # Export CSV
        csv_path = f"{base_path}.csv"
        df = pd.DataFrame([{'x_local': p['x'], 'y_local': p['y']} for p in self.points])
        df.to_csv(csv_path, index=False)

        messagebox.showinfo("Export Successful",
                            f"Files exported:\n{png_path}\n{csv_path}\n\nTotal points: {len(self.points)}")


def main():
    root = tk.Tk()
    app = ClusterPointGenerator(root)
    root.mainloop()


if __name__ == '__main__':
    main()