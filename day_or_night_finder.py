import os
import shutil
import random
from PIL import Image
import numpy as np
from multiprocessing import Pool

input = r"D:\Medias\fotograflar_Alanya_Otopark_Foto"
day_photos = r"D:\Medias\fotograflar_Karasu_Belediyesi_Foto\day_photos\03"
night_photos = r"D:\Medias\fotograflar_Karasu_Belediyesi_Foto\night_photos"

os.makedirs(day_photos, exist_ok=True)
os.makedirs(night_photos, exist_ok=True)

max_brightness = 80

def get_brightness(dosya):
    file_path = os.path.join(input, dosya)
    try:
        with Image.open(file_path) as img:
            img = img.convert("L").resize((64, 64))
            average_brightness = np.mean(img)

        if average_brightness > max_brightness:
            pathh = os.path.join(day_photos, dosya)
        else:
            pathh = os.path.join(night_photos, dosya)

        shutil.move(file_path, pathh)
        print(file_path)
    except Exception as e:
        print(f"error : {dosya}, {e}")

def main():
    dosyalar = [f for f in os.listdir(input) if os.path.isfile(os.path.join(input, f))]
    random.shuffle(dosyalar)

    with Pool(processes=os.cpu_count()) as p:
        p.map(get_brightness, dosyalar)

if __name__ == "__main__":
    main()
