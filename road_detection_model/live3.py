from ultralytics import RTDETR
import cv2
import cvzone
import math
from playsound import playsound
import winsound
import threading
import time
import os

last_alarm_time = 0

def play_alarm():
    global last_alarm_time
    sound_path = "warning_sound/warning.mp4"

    if not os.path.exists(sound_path):
        print(f"Sound file not found: {sound_path}")
        return

    if time.time() - last_alarm_time > 2:
        last_alarm_time = time.time()
        try:
            playsound(os.path.abspath(sound_path))
        except Exception:
            try:
                winsound.Beep(1000, 500)
            except Exception:
                pass


def live_predict(model_path, setting, wait_key, classNames, video_path=None):

    if setting == 'live':
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
    elif setting == 'static':
        if video_path is None:
            raise ValueError("video_path required for static mode")
        cap = cv2.VideoCapture(video_path)
    else:
        raise ValueError("setting must be 'live' or 'static'")

    # 🔁 LOAD RT-DETR MODEL
    model = RTDETR(model_path)

    classColors = {
        "different traffic sign": (255, 100, 50),
        "pedestrian": (128, 0, 128),
        "car": (0, 255, 0),
        "truck": (255, 165, 0),
        "warning sign": (0, 255, 255),
        "prohibition sign": (0, 0, 255),
        "pedestrian crossing": (173, 216, 230),
        "speed limit sign": (255, 192, 203)
    }

    fallback_colors = [(0, 100, 0), (255, 255, 0)]
    for i, cls in enumerate(classNames):
        if cls not in classColors:
            classColors[cls] = fallback_colors[i % len(fallback_colors)]

    while True:
        success, img = cap.read()
        if not success:
            break

        # 🔁 RT-DETR INFERENCE
        results = model(img)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls = classNames[cls_id]
                conf = round(float(box.conf[0]), 2)

                color = classColors.get(cls, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cvzone.putTextRect(
                    img, f"{cls}",
                    (max(0, x1), max(35, y1)),
                    scale=1.5, thickness=2,
                    offset=3, colorR=color, colorT=(0, 0, 0)
                )

                h, w, _ = img.shape
                obj_cx = (x1 + x2) // 2
                obj_cy = y2
                screen_cx = w // 2
                screen_cy = h - 1

                dist_pixels = math.hypot(screen_cx - obj_cx, screen_cy - obj_cy)

                if cls in ["car", "truck", "bus", "motorcycle", "pedestrian", "pedestrian crossing"]:
                    cv2.line(img, (screen_cx, screen_cy), (obj_cx, obj_cy), color, 2)
                    cvzone.putTextRect(
                        img, f"{int(dist_pixels)} px",
                        (x1, y2 - 10),
                        scale=1.2, thickness=2,
                        offset=3, colorR=color, colorT=(0, 0, 0)
                    )

                    if dist_pixels < 300:
                        cvzone.putTextRect(
                            img, "WARNING: TOO CLOSE",
                            (int(w * 0.2), int(h * 0.2)),
                            scale=3, thickness=3,
                            colorR=(0, 0, 255)
                        )
                        threading.Thread(target=play_alarm, daemon=True).start()

        cv2.imshow("RT-DETR Detection", img)

        if cv2.waitKey(wait_key) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    class_names_finetuned = [
        "car", "different traffic sign", "green traffic light",
        "motorcycle", "pedestrian", "pedestrian crossing",
        "prohibition sign", "red traffic light",
        "speed limit sign", "truck", "warning sign"
    ]

    live_predict(
        model_path="rtdetr-l.pt",
        setting="static",
        wait_key=5,
        classNames=class_names_finetuned,
        video_path="test_images/test_film.mp4"
    )
