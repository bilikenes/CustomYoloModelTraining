import os
import shutil

# Klasör yolları
klasor1 = r"C:\Users\PC\Desktop\dataset_for_YOLO\valid\images"   # ilk klasör
klasor2 = r"C:\Users\PC\Desktop\plates_from_Karasu\fotograflar_Karasu_Belediyesi_Foto"   # ikinci klasör
hedef   = r"C:\Users\PC\Desktop\plates_from_Karasu\valid"   # yeni klasör

# Hedef klasör yoksa oluştur
os.makedirs(hedef, exist_ok=True)

# Klasör1'deki dosya isimleri
dosyalar1 = set(os.listdir(klasor1))
dosyalar2 = set(os.listdir(klasor2))

# İki klasörde ortak olan dosyalar
ortak_dosyalar = dosyalar1.intersection(dosyalar2)

print(f"{len(ortak_dosyalar)} dosya bulundu.")

# Ortak dosyaları klasör2'den hedef klasöre kopyala
for dosya in ortak_dosyalar:
    kaynak = os.path.join(klasor2, dosya)
    hedef_yol = os.path.join(hedef, dosya)
    shutil.copy2(kaynak, hedef_yol)

print("Kopyalama işlemi tamamlandı.")
