import h5py
import json

file_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ005647_u673_1_40X\cd8.hdf5"
# Open your h5py file
with h5py.File(file_path, 'r') as f:
    coords = []

    # Iterate through all tiles in wsi_cells
    for key in f['wsi_cells'].keys():
        if key.startswith('tile_'):
            # Read tile data
            data = f['wsi_cells'][key][()]

            # Decode if bytes
            if isinstance(data, bytes):
                data = data.decode('utf-8')

            # Parse JSON
            geojson = json.loads(data)

            # Extract coordinates for label 2
            for feature in geojson['features']:
                if feature['properties']['label'] == 2:
                    coord = feature['geometry']['coordinates']
                    coords.append(coord)

    # Print results
    print(f"Total points with label 2: {len(coords)}\n")
    for coord in coords:
        print(f"{coord[0]}, {coord[1]}")