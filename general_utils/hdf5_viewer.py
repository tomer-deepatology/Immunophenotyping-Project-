import h5py

# file_path = r"C:\Users\tomer\Downloads\2bab21b6-742b-4e88-91df-784900b81214\mask_file.hdf5"
file_path = "/mnt/c/Users/tomer/Desktop/data/project 1/225_panCK CD8_TRSPZ005647_u673_1_40X/cd8.hdf5"
with h5py.File(file_path, 'r') as f:
    def print_contents(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\n{name}:")
            print(f"  Shape: {obj.shape}, Type: {obj.dtype}")

            # # Handle scalar datasets differently
            # if obj.shape == ():
            #     print(f"  Data: {obj[()]}")
            # else:
            #     print(f"  Data:\n{obj[:]}")


    f.visititems(print_contents)