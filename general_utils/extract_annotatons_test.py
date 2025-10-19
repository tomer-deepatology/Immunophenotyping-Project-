import h5py
import json

annotations_path = r"C:\Users\tomer\Desktop\data\project 1\225_panCK CD8_TRSPZ008500_u673_1_40X_report\c6e89ee9-3ce1-4b69-b3a6-953e950fbec5\mask_file.hdf5"


def explore_h5_structure(name, obj):
    """Recursively explore HDF5 file structure"""
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset: shape={obj.shape}, dtype={obj.dtype}")
        # If it's a small dataset, print a sample
        if obj.size < 10:
            print(f"  Sample data: {obj[()]}")


with h5py.File(annotations_path, 'r') as f:
    print("=== Full HDF5 Structure ===\n")
    f.visititems(explore_h5_structure)

    print("\n=== Top-level keys ===")
    print(list(f.keys()))

    # Check for common annotation storage locations
    print("\n=== Checking for annotation groups ===")

    if 'annotations' in f:
        print("\nFound 'annotations' group:")
        print(list(f['annotations'].keys()))

    if 'polygons' in f:
        print("\nFound 'polygons' group:")
        print(list(f['polygons'].keys()))

    if 'manual_annotations' in f:
        print("\nFound 'manual_annotations' group:")
        print(list(f['manual_annotations'].keys()))

    # Look for any groups containing 'Tissue' or 'Tumor'
    print("\n=== Searching for Tissue/Tumor annotations ===")
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            subkeys = list(f[key].keys())
            print(f"\nGroup '{key}' contains: {subkeys}")

            # Check if any subkeys contain polygon-like data
            for subkey in subkeys:
                if 'tissue' in subkey.lower() or 'tumor' in subkey.lower():
                    print(f"  Found potential annotation: {subkey}")
                    try:
                        data = f[f'{key}/{subkey}'][()]
                        print(f"    Shape: {data.shape}, dtype: {data.dtype}")
                        if data.dtype == 'object' or 'S' in str(data.dtype):
                            # Might be JSON
                            sample = json.loads(data) if isinstance(data, (str, bytes)) else data
                            print(f"    Sample: {str(sample)[:200]}")
                    except Exception as e:
                        print(f"    Error reading: {e}")