

"""import os
import shutil

# Ana klasörün yolu (sen kendi yolunu yaz)
ana_klasor = "C:/Users/PC/Desktop/plates/rectangle_plates/01"
hedef_klasor = "C:/Users/PC/Desktop/plates/dataset/train/labels"


# Hedef klasör yoksa oluştur
os.makedirs(hedef_klasor, exist_ok=True)

# Ana klasördeki alt klasörleri gez
for klasor in os.listdir(ana_klasor):
    alt_klasor_yolu = os.path.join(ana_klasor, klasor)
    
    # Sadece klasörleri kontrol et
    if os.path.isdir(alt_klasor_yolu):
        labels_klasoru = os.path.join(alt_klasor_yolu, "labels")
        
        if os.path.exists(labels_klasoru):
            for dosya in os.listdir(labels_klasoru):
                kaynak_dosya = os.path.join(labels_klasoru, dosya)
                hedef_dosya = os.path.join(hedef_klasor, f"{klasor}_{dosya}")
                
                # Aynı isimli dosyaların üzerine yazılmasını engellemek için
                shutil.copy2(kaynak_dosya, hedef_dosya)
                print(f"Kopyalandı: {kaynak_dosya} -> {hedef_dosya}")

print("Tüm label dosyaları toplandı ✅")
"""

"""import os
import shutil

# 1-28 arasındaki klasörlerin bulunduğu ana klasör
ana_klasor = "C:/Users/PC/Desktop/plates/rectangle_plates/01"

# Kaynak klasör (aynı isimli dosyaları buradan alacağız)
kaynak_klasor = "C:/Users/PC/Desktop/plates/01"

# Çıkış klasörü
hedef_klasor = "C:/Users/PC/Desktop/plates/dataset/train/images"

# Hedef klasörü oluştur
os.makedirs(hedef_klasor, exist_ok=True)

# 1–28 arası klasörleri gez
for i in range(9, 29):
    klasor_adi = f"rectangle_plates_01_{i:02d}"   # örn: rectangle_plates_01_09
    alt_klasor = os.path.join(ana_klasor, klasor_adi)

    if not os.path.exists(alt_klasor):
        continue

    # klasörün içindeki tüm dosyaları al, labels klasörünü atla
    dosya_listesi = [d for d in os.listdir(alt_klasor) if d.lower() != "labels"]

    # her dosya için
    for dosya in dosya_listesi:
        kaynak_dosya = None

        # diğer yapıda aynı isimli dosyayı ara
        for j in range(9, 29):
            diger_klasor = os.path.join(kaynak_klasor, f"{j:02d}")
            aday_dosya = os.path.join(diger_klasor, dosya)

            if os.path.exists(aday_dosya):
                kaynak_dosya = aday_dosya
                break
        
        # Eşleşme bulunduysa kopyala
        if kaynak_dosya:
            hedef_dosya = os.path.join(hedef_klasor, f"{klasor_adi}_{dosya}")
            shutil.copy2(kaynak_dosya, hedef_dosya)
            print(f"Kopyalandı: {kaynak_dosya} -> {hedef_dosya}")"""


"""import torch

print(torch.cuda.is_available())  # True ise GPU kullanılabilir
print(torch.cuda.get_device_name(0))  # GPU ismini gösterir
"""

import os

# Buraya klasör yolunu yaz
klasor_yolu = r"C:\Users\PC\Desktop\plates\07\31"   # Örn: Windows

for dosya in os.listdir(klasor_yolu):
    if dosya.startswith("Okunamadı"):
        dosya_yolu = os.path.join(klasor_yolu, dosya)
        if os.path.isfile(dosya_yolu):  # sadece dosyaları sil
            os.remove(dosya_yolu)
            print(f"Silindi: {dosya_yolu}")

print("ok")