import os

txt_folder = r"D:\Medias\plates\rectangle_plates\rectangle_plates_01_31\labels"
img_folder = r"D:\Medias\plates\01\31"

txt_names = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.endswith(".txt")}

for img_file in os.listdir(img_folder):
    if img_file.lower().endswith(".jpg"):
        img_name = os.path.splitext(img_file)[0]
        if img_name not in txt_names:
            img_path = os.path.join(img_folder, img_file)
            print("Siliniyor:", img_path)
            os.remove(img_path)
