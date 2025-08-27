import os

# Buraya resimlerin bulunduğu klasörü yaz
klasor = r"C:\Users\PC\Desktop\dataset_for_YOLO\train\images"

# Klasördeki tüm dosyaları listele
dosyalar = os.listdir(klasor)

for dosya in dosyalar:
    # Sadece .png dosyalarını kontrol et
    if dosya.lower().endswith(".png"):
        png_yolu = os.path.join(klasor, dosya)
        txt_yolu = png_yolu + ".txt"  # Aynı isimle .txt dosyası
        # Eğer .txt yoksa resmi sil
        if not os.path.exists(txt_yolu):
            print(f"Siliniyor: {png_yolu}")
            os.remove(png_yolu)

print("İşlem tamamlandı.")
