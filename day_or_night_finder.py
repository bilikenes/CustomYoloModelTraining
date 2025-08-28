import os
import shutil
from PIL import Image
import numpy as np
from multiprocessing import Pool

girdi_klasoru = r"C:\Users\PC\Desktop\test"
gunduz_klasoru = r"C:\Users\PC\Desktop\test\gunduz"
gece_klasoru = r"C:\Users\PC\Desktop\test\gece"

os.makedirs(gunduz_klasoru, exist_ok=True)
os.makedirs(gece_klasoru, exist_ok=True)

ESIK = 80

def isle(dosya):
    yol = os.path.join(girdi_klasoru, dosya)
    try:
        with Image.open(yol) as img:

            img = img.convert("L").resize((64, 64))
            ortalama_parlaklik = np.mean(img)

        if ortalama_parlaklik > ESIK:
            hedef = os.path.join(gunduz_klasoru, dosya)
        else:
            hedef = os.path.join(gece_klasoru, dosya)

        shutil.move(yol, hedef)
    except Exception as e:
        print(f"Hata: {dosya}, {e}")

def main():
    dosyalar = [f for f in os.listdir(girdi_klasoru) if os.path.isfile(os.path.join(girdi_klasoru, f))]

    with Pool(processes=os.cpu_count()) as p:
        p.map(isle, dosyalar)

if __name__ == "__main__":
    main()
