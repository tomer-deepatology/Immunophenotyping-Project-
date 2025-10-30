import openslide

file_path = r"C:\Users\tomer\Desktop\data\demo_sunday\OS-2.ndpi"

slide = openslide.OpenSlide(file_path)

print(f"Total levels: {slide.level_count}")
print()

for level in range(slide.level_count):
    width, height = slide.level_dimensions[level]
    print(f"Level {level}: {width} x {height}")

slide.close()