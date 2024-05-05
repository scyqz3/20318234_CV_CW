from moviepy.editor import VideoFileClip
import os


def convert_to_mp4(video_path):
    # Converts a WindowsPath object to a string
    video_path_str = str(video_path)

    # Get file name and extension
    file_name, extension = os.path.splitext(video_path_str)

    # Define the output MP4 file path
    mp4_output_path = file_name + ".mp4"

    # Use MoviePy to convert video formats
    clip = VideoFileClip(video_path_str)
    clip.write_videofile(mp4_output_path)

    return mp4_output_path
