# import os

# klasor = r"D:\Medias\dataset_for_train_YOLO_1\train\images"
# klasor_2 = r"D:\Medias\dataset_for_train_YOLO_1\train\labels"

# dosyalar = os.listdir(klasor)
# txt_dosyalari = os.listdir(klasor_2)

# for dosya in dosyalar:
#     is_ok = False
#     if dosya.lower().endswith(".png"):
#         real_png = os.path.join(klasor, dosya)
#         txt_file = os.path.join(klasor, dosya)
#         txt_name = os.path.basename(txt_file)
#         txt_name = "D:/Medias/dataset_for_train_YOLO_1/train/labels" + txt_name[:-4] + ".txt"
        
#         for txt in txt_dosyalari:
#             txt_path = os.path.join(klasor_2, txt)
#             print(txt_path)
#             if(txt_path == txt_name):
#                 print("var")
#                 is_ok = True
#                 break
#         if(is_ok == False):
#             pass
#             #os.remove(real_png)

#         # png_yolu = png_yolu[:-4] + ".txt"
#         # print(png_yolu)
#         # print("a--"+real_png)

#         # if png_yolu not in dosyalar:
#         #     print(f"deleting : {real_png}")
#         #     os.remove(real_png)


#     # for txt in txt_dosyaları:
#     #     png_yolu = png_yolu + ".txt"
#     #     print(png_yolu)
#         # txt_yolu = os.path.join(klasor_2, txt)
#         # txt_yolu = txt_yolu[:-4]
    


#     # elif dosya.lower().endswith(".txt"):
#     #     txt_yolu = os.path.join(klasor, dosya)
#     #     print("test")
#     #     print(txt_yolu[:-4])
#     #     if not os.path.exists(txt_yolu):
#     #         print("test1")
#     #         print(f"deleting : {png_yolu}")
#     #         os.remove(png_yolu)

# print("ok.")

