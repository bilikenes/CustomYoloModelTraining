# from ultralytics import YOLO
# import cv2
# import os
# import csv

# model = YOLO(r"runs\detect\train4\weights\best.pt")
# video_path = r"D:\Medias\0821.mp4"

# save_dir = "detected_plates"
# os.makedirs(save_dir, exist_ok=True)

# csv_file = os.path.join(save_dir, "detections.csv")
# with open(csv_file, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["filename", "class_id", "confidence", "time_sec", "x1", "y1", "x2", "y2"])

# cap = cv2.VideoCapture(video_path)
# frame_count = 0
# save_count = 0
# frame_skip = 2

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     frame_resized = cv2.resize(frame, (1280, 720)) 

#     results = model.predict(frame_resized, conf=0.50, verbose=False)
#     current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#     print(f"Anlık zaman: {current_time_sec:.2f} saniye")

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             if cls == 0 and conf > 0.25:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 plate_img = frame_resized[y1:y2, x1:x2]

#                 filename = f"{save_dir}/plate_{save_count}_{int(current_time_sec*1000)}ms.jpg"
#                 cv2.imwrite(filename, plate_img)

#                 with open(csv_file, mode="a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([filename, cls, conf, current_time_sec, x1, y1, x2, y2])

#                 print(f"[INFO] {current_time_sec:.2f}s → Plaka kaydedildi: {filename}")
#                 save_count += 1

# cap.release()
# cv2.destroyAllWindows()
# print("ok")

# from ultralytics import YOLO
# import cv2
# import os
# import csv

# model = YOLO(r"runs\detect\train4\weights\best.pt")
# video_path = r"D:\Medias\0821_2.mp4"

# save_dir = "detected_plates"
# os.makedirs(save_dir, exist_ok=True)

# csv_file = os.path.join(save_dir, "detections.csv")
# with open(csv_file, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["filename", "class_id", "confidence", "time_sec", "x1", "y1", "x2", "y2"])

# cap = cv2.VideoCapture(video_path)
# frame_count = 0
# save_count = 0
# frame_skip = 2

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     frame_resized = cv2.resize(frame, (1280, 720)) 

#     results = model.predict(frame_resized, conf=0.50, verbose=False)
#     current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#     print(f"Anlık zaman: {current_time_sec:.2f} saniye")

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             if cls == 0 and conf > 0.25:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                
#                 cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame_resized, f"Plate {conf:.2f}", 
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             0.6, (0, 255, 0), 2)

#                 plate_img = frame_resized[y1:y2, x1:x2]
#                 filename = f"{save_dir}/plate_{save_count}_{int(current_time_sec*1000)}ms.jpg"
#                 cv2.imwrite(filename, plate_img)

#                 with open(csv_file, mode="a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([filename, cls, conf, current_time_sec, x1, y1, x2, y2])

#                 print(f"[INFO] {current_time_sec:.2f}s → Plaka kaydedildi: {filename}")
#                 save_count += 1

#     cv2.imshow("Detected Plates", frame_resized)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("ok")

from ultralytics import YOLO
import cv2
import os
import csv

model = YOLO(r"runs\detect\train4\weights\best.pt")
video_path = r"C:\Users\PC\Desktop\plates\0821_1.mp4"

save_dir = "detected_plates"
os.makedirs(save_dir, exist_ok=True)

csv_file = os.path.join(save_dir, "detections.csv")
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class_id", "confidence", "time_sec", "x1", "y1", "x2", "y2"])

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
save_count = 0
frame_skip = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    current_time_sec = frame_count / fps
    print(f"Anlık zaman: {current_time_sec:.2f} saniye")

    frame_resized = cv2.resize(frame, (1280, 720)) 

    results = model.predict(frame_resized, conf=0.50, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Plate {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                plate_img = frame_resized[y1:y2, x1:x2]
                filename = f"{save_dir}/plate_{save_count}_{int(current_time_sec*1000)}ms.jpg"
                cv2.imwrite(filename, plate_img)

                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, cls, conf, current_time_sec, x1, y1, x2, y2])

                print(f"[INFO] {current_time_sec:.2f}s → Plaka kaydedildi: {filename}")
                save_count += 1

    cv2.imshow("Detected Plates", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("ok")