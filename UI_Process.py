import Video_process
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds
import Video_Convert
import gradio as gr


# Start processing
def start_processing(input_path, output_path, detect_class, model, progress=gr.Progress(track_tqdm=True)):
    global should_continue
    should_continue = True

    detect_class = int(detect_class)
    model = YOLO(model)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    output_video_path = Video_process.detect_and_track(input_path, output_path, detect_class, model, tracker)

    # Convert to MP4 format
    mp4_output_path = Video_Convert.convert_to_mp4(output_video_path)

    return mp4_output_path, mp4_output_path


# Stop processing
def stop_processing():
    global should_continue
    should_continue = False
    return "Attempt to terminate processing..."
