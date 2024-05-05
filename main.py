import Video_process
from ultralytics import YOLO
import os
import deep_sort.deep_sort.deep_sort as ds


if __name__ == "__main__":
    # Assign input video path.
    input_path = "mytest03.mp4"

    # Set output directory to the existing dir: "output"
    current_directory = os.getcwd()
    output_path = os.path.join(current_directory, 'output')

    # Load yoloV8 Model weights
    model = YOLO("yolov8x.pt")

    # Set the target category to be detected and tracked
    # The first category of the official yoloV8 model is 'person',
    # so we set the detect_class value to 0
    detect_class = 0
    # model.names Returns all object classes supported by the model
    print(f"detecting {model.names[detect_class]}")

    # Load Model--DeepSort
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    Video_process.detect_and_track(input_path, output_path, detect_class, model, tracker)
