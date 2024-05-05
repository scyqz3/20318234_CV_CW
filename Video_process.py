import Detection_module
import cv2
from pathlib import Path
import numpy as np

def adjust_contrast_brightness(frame, contrast, brightness):
    adjusted_frame = np.clip(contrast * frame + brightness, 0, 255)
    adjusted_frame = cv2.convertScaleAbs(adjusted_frame)
    return adjusted_frame

def adjust_saturation(frame, saturation):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] * saturation, 0, 255)  # Modify saturation channel
    adjusted_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)  # Convert HSV back to BGR
    return adjusted_frame

def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    Process a video, detect and track objects.
    - input_path: Path to the input video file.
    - output_path: Path to save the processed video.
    - detect_class: Index of the target class to detect and track.
    - model: Model used for object detection.
    - tracker: Model used for object tracking.
    """

    # Open the video file using OpenCV.
    cap = cv2.VideoCapture(input_path)

    # Check if the video file was opened successfully.
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    # Get the video frame rate.
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the video resolution (width and height).
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Set the output video save path.
    output_video_path = Path(output_path) / "output.avi"

    # Set the video codec as XVID for AVI files.
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Create a VideoWriter object to write the video.
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    # Process each frame of the video.
    while True:
        success, frame = cap.read()  # Read the video frame by frame.

        # Exit the loop if the read process failed or the video processing finished.
        if not success:
            break

            # Adjust contrast, brightness, and saturation of the frame.
        contrast = 0.85  # Example: Increase contrast by 20%
        brightness = 15  # Example: Increase brightness by 10 units
        saturation = 0.95  # Example: No change in saturation

        frame = adjust_contrast_brightness(frame, contrast, brightness)
        frame = adjust_saturation(frame, saturation)

        # Use the YOLOv8 model to detect objects in the current frame.
        results = model(frame, stream=True)

        # Extract detection information from the prediction results.
        detections, confarray = Detection_module.Extract_DetInfo(results, detect_class)

        # Use the deepsort model to track the detected objects.
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert the position information to integers.

            # Draw bounding box and text.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            Detection_module.Text_Draw_with_BG(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5,
                              text_color=(255, 255, 255), bg_color=(0, 0, 255))

        # Write the processed frame to the output video file.
        output_video.write(frame)

    output_video.release()  # Release the VideoWriter object.
    cap.release()

    print(f'Output directory: {output_video_path}')
    return output_video_path