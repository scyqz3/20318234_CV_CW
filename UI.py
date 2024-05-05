from ultralytics import YOLO
import tempfile
import gradio as gr
import UI_Process

# Control whether the process is terminated
should_continue = True


def get_classes_detected_objects(model_file):
    Current_model = YOLO(model_file)
    class_names = list(Current_model.names.values())  # Get the list of category names directly
    del Current_model  # Delete model instances to free resources
    return class_names


if __name__ == "__main__":
    # YoloV8, V9 official model list
    model_list = ["yolov9c.pt", "yolov9e", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

    # Gets all the categories that the YoloV8 model can detect, by default calling the first model in model_list
    detect_classes = get_classes_detected_objects(model_list[0])

    # gradio Input example: Video file path, a randomly generated output directory, category to detect, model to use
    examples = [
        ["mytest03.mp4", tempfile.mkdtemp(), detect_classes[0], model_list[0]],
        ["mytest02.mp4", tempfile.mkdtemp(), detect_classes[0], model_list[0]],
        ["mytest01.mp4", tempfile.mkdtemp(), detect_classes[0], model_list[0]]
    ]

    # Use Gradio's Blocks to create a GUI

    with gr.Blocks() as demo:
        with gr.Tab("Object Tracking Tool"):
            # Use Markdown to display text information that describes the functionality of the interface
            gr.HTML("""
                <div style="text-align: center;">
                    <h1>Person Tracking from Videos</h1>
                    <p>Main techniques: Yolo && DeepSort</p>
                </div>
            """)
            with gr.Row():
                with gr.Column():
                    input_path = gr.Video(label="Input video")
                    model = gr.Dropdown(model_list, value=0, label="Model")
                    detect_class = gr.Dropdown(detect_classes, value=0, label="Class",
                                               type='index')
                    output_dir = gr.Textbox(label="Output dir",
                                            value=tempfile.mkdtemp())
                    with gr.Row():
                        start_button = gr.Button("Start Processing")
                        stop_button = gr.Button("Stop")
                with gr.Column():
                    output = gr.Video()
                    output_path = gr.Textbox(label="Output path")

                    gr.Examples(examples, label="Examples",
                                inputs=[input_path, output_dir, detect_class, model],
                                outputs=[output, output_path],
                                fn=UI_Process.start_processing,
                                cache_examples=False)

        # Combine the functions with the button
        start_button.click(UI_Process.start_processing, inputs=[input_path, output_dir, detect_class, model],
                           outputs=[output, output_path])
        stop_button.click(UI_Process.stop_processing)

    demo.launch()
