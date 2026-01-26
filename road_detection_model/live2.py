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
    # Check if file exists
    sound_path = "warning_sound/warning.mp4"
    if not os.path.exists(sound_path):
        print(f"Sound file not found: {sound_path}")
        return

    if time.time() - last_alarm_time > 2: # 2 second cooldown (MP4 might be longer)
        last_alarm_time = time.time()
        try:
            # Try playing the custom sound
            abs_path = os.path.abspath(sound_path)
            playsound(abs_path)
        except Exception as e:
            print(f"Error playing sound with playsound: {e}")
            print("Falling back to system beep.")
            # Fallback to system beep: 1000Hz for 500ms
            try:
                winsound.Beep(1000, 500)
            except Exception as e_beep:
                print(f"Error playing system beep: {e_beep}")


def live_predict(model_path, setting, wait_key, classNames, video_path=None):
    """
    Perform live object detection using YOLO model.

    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
    - wait_key (int): Time in milliseconds to wait between frames. A value of 0 means wait indefinitely.
    - classNames (list of str): List of class names that the model has been trained to recognize.
    - video_path (str, optional): Path to the video file for 'static' setting. Required if setting is 'static'.

    Raises:
    - ValueError: If 'setting' is not 'live' or 'static', or if 'video_path' is not provided for 'static' setting.
    """

    # Initialize video capture based on the setting
    if setting == 'live':
        # For live webcam feed
        cap = cv2.VideoCapture(1)  # Open default webcam
        cap.set(3, 640)  # Set the width of the frame to 640 pixels
        cap.set(4, 480)  # Set the height of the frame to 480 pixels
    elif setting == 'static':
        # For video file
        if video_path is None:
            raise ValueError("In 'static' setting you must pass video_path")
        cap = cv2.VideoCapture(video_path)  # Load video file
    else:
        # Raise an error if setting is invalid
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Define specific colors for selected classes
    classColors = {
        "different traffic sign": (255, 100, 50),  # Blue
        "pedestrian": (128, 0, 128),  # Purple
        "car": (0, 255, 0),  # Green
        "truck": (255, 165, 0),  # Orange
        "warning sign": (0, 255, 255),  # Yellow
        "prohibition sign": (0, 0, 255),  # Red
        "pedestrian crossing": (173, 216, 230),  # Light Blue
        "speed limit sign": (255, 192, 203)  # Pink
    }

    # Define colors for remaining classes
    remaining_colors = {
        "dark green": (0, 100, 0),  # Dark Green
        "dark yellow": (255, 255, 0)  # Dark Yellow
    }

    # Assign colors to the remaining classes
    remaining_color_list = list(remaining_colors.values())
    for i, cls in enumerate(classNames):
        if cls not in classColors:
            classColors[cls] = remaining_color_list[i % len(remaining_color_list)]

    while True:
        # Read a frame from the video capture
        success, img = cap.read()
        if not success:
            break  # End of video or cannot read frame

        # Perform object detection on the current frame
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            current_objects = [] # Objects in current frame
            for box in boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the color for the bounding box based on the detected class
                cls = classNames[int(box.cls[0])]
                color = classColors.get(cls, (255, 255, 255))  # Default to white if class not found in color map

                # Draw a thin rectangle around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Thickness set to 2 for thin rectangles

                # Calculate the confidence score and format it
                conf = math.floor(box.conf[0] * 100) / 100

                # Display class name and confidence score
                cvzone.putTextRect(img, f"{cls}", (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))

                # Calculate Distance from Center-Bottom of Screen (Car Hood) to Object
                height, width, _ = img.shape
                
                # Object bottom center
                obj_cx = (x1 + x2) // 2
                obj_cy = y2
                
                # Screen bottom center (always inside the image)
                screen_cx = width // 2
                screen_cy = height - 1
                
                # Euclidean Distance
                dist_pixels = math.hypot(screen_cx - obj_cx, screen_cy - obj_cy)
                
                # Filter for Vehicles and Pedestrians
                if cls in ["car", "truck", "bus", "motorcycle", "pedestrian", "pedestrian crossing"]:
                    # Draw Line from Bottom Center to Object
                    cv2.line(img, (screen_cx, screen_cy), (obj_cx, obj_cy), color, 2)
                    cvzone.putTextRect(img, f"{int(dist_pixels)} px", (max(0, x1), y2 - 10), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))
                    
                    # ALERT LOGIC
                    if dist_pixels < 300: # Threshold for alert
                         cvzone.putTextRect(img, "WARNING: TOO CLOSE", (int(width*0.2), int(height*0.2)), scale=3, thickness=3, colorR=(0, 0, 255))
                         threading.Thread(target=play_alarm, daemon=True).start()


        # Display the resulting frame in a window
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(wait_key) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define class names for different settings
    class_names_pretrained = [
        "person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow",
        "tl_none", "traffic sign", "train", "tl_green"
    ]

    class_names_finetuned = [
        "car", "different traffic sign", "green traffic light", "motorcycle", "pedestrian", "pedestrian crossing",
        "prohibition sign", "red traffic light", "speed limit sign", "truck", "warning sign"
    ]

    # Run the live_predict function with the fine-tuned model and specified settings
    live_predict(
        model_path='Models/fine_tuned_yolov8s.pt',
        setting='live',
        wait_key=5,
        classNames=class_names_finetuned,
        video_path='test_images/long_video1.mp4'
    )


