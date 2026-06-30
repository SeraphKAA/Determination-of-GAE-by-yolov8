from ultralytics import YOLO
import cv2

def res(model_name: str, image_name: str):
    model = YOLO(f"app/{model_name}.pt")
    
    image_path = f"app/{image_name}.png"
    results = model(image_path, conf=0.25)
    result_img = results[0].plot()

    cv2.imwrite(f"app/result_{model_name}.png", result_img)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]

        print(
            f"Класс: {model.names[cls]}, "
            f"Уверенность: {conf:.2f}, "
            f"Координаты: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
        )



# model = YOLO("app/best_ethnos.pt")
# results = model(image_path, conf=0.25)
# result_img = results[0].plot()

# cv2.imwrite("app/result.png", result_img)
# for box in results[0].boxes:
#     cls = int(box.cls[0])
#     conf = float(box.conf[0])

#     x1, y1, x2, y2 = box.xyxy[0]

#     print(
#         f"Класс: {model.names[cls]}, "
#         f"Уверенность: {conf:.2f}, "
#         f"Координаты: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
#     )


res("best_gender",  "1")
print("=" * 30)
res("best_age", "1")
print("=" * 30)
res("best_ethnos", "1")