from ultralytics import YOLO
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
        except:
            try:
                winsound.Beep(1000, 500)
            except:
                pass


def live_predict(model_path, setting, wait_key, classNames, video_path=None):

    if setting == 'live':
        cap = cv2.VideoCapture(1)
        cap.set(3, 640)
        cap.set(4, 480)
    elif setting == 'static':
        cap = cv2.VideoCapture(video_path)
    else:
        raise ValueError("setting must be 'live' or 'static'")

    model = YOLO(model_path)

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

    while True:
        frame_start = time.perf_counter()

        success, img = cap.read()
        if not success:
            break

        # ================= YOLO INFERENCE =================
        results = model(img, stream=True)

        inference_ms = 0

        for r in results:
            if hasattr(r, "speed") and "inference" in r.speed:
                inference_ms = r.speed["inference"]

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = classNames[int(box.cls[0])]
                color = classColors.get(cls, (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cvzone.putTextRect(
                    img, cls, (x1, max(35, y1)),
                    scale=1.2, thickness=2, offset=3,
                    colorR=color, colorT=(0, 0, 0)
                )

                h, w, _ = img.shape
                obj_cx = (x1 + x2) // 2
                obj_cy = y2
                screen_cx = w // 2
                screen_cy = h

                dist = math.hypot(screen_cx - obj_cx, screen_cy - obj_cy)

                if cls in ["car", "truck", "pedestrian", "motorcycle"]:
                    cv2.line(img, (screen_cx, screen_cy), (obj_cx, obj_cy), color, 2)
                    cvzone.putTextRect(
                        img, f"{int(dist)} px",
                        (x1, y2 - 10),
                        scale=1.2, thickness=2, offset=3
                    )

                    if dist < 150:
                        cvzone.putTextRect(
                            img, "WARNING: TOO CLOSE",
                            (int(w * 0.25), int(h * 0.2)),
                            scale=3, thickness=3,
                            colorR=(0, 0, 255)
                        )
                        threading.Thread(target=play_alarm, daemon=True).start()

        # ================= METRICS =================
        latency_ms = (time.perf_counter() - frame_start) * 1000
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        cvzone.putTextRect(
            img,
            f"Latency: {latency_ms:.1f} ms",
            (10, 40),
            scale=1.3, thickness=2
        )

        cvzone.putTextRect(
            img,
            f"Inference: {inference_ms:.1f} ms",
            (10, 80),
            scale=1.3, thickness=2
        )

        cvzone.putTextRect(
            img,
            f"FPS: {fps:.1f}",
            (10, 120),
            scale=1.3, thickness=2
        )

        cv2.imshow("YOLO Real-Time Detection", img)

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
        model_path="Models/fine_tuned_yolov8s.pt",
        setting="static",
        wait_key=5,
        classNames=class_names_finetuned,
        video_path="test_images/test_film.mp4"
    )
