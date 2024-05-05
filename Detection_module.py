import numpy as np
import cv2


# Draw text with a background.
def Text_Draw_with_BG(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                      text_color=(255, 0, 255), bg_color=(0, 0, 0), thickness=3):
    """
    Draw text with a background on the image.
    :param img: Input image.
    :param text: Text to be drawn.
    :param origin: Origin point (bottom left corner) of the text.
    :param font: Font type.
    :param font_scale: Font scale factor.
    :param text_color: Text color.
    :param bg_color: Background color.
    :param thickness: Thickness of the text.
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)

    # Calculate the size of the combined text
    (id_text_width, id_text_height), _ = cv2.getTextSize("ID:", font, font_scale, thickness)

    # Calculate the size of the background rectangle
    bg_width = text_width + id_text_width + 20
    bg_height = max(text_height, id_text_height) + 20

    # Calculate the bottom left and top right coordinates of the background rectangle
    bg_bottom_left = origin
    bg_top_right = (origin[0] + bg_width, origin[1] - bg_height)

    # Draw the background rectangle
    cv2.rectangle(img, bg_bottom_left, bg_top_right, bg_color, -1)

    # Calculate the origin for the "ID" text
    id_text_origin = (origin[0] + 10, origin[1] - bg_height // 2 + id_text_height // 2)

    # Calculate the origin for the original text
    text_origin = (origin[0] + id_text_width + 20, origin[1] - bg_height // 2 + text_height // 2)

    # Draw "ID" text inside the rectangle
    cv2.putText(img, "ID:", id_text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    # Draw the original text inside the rectangle
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


# Extract and process detection information from model results
def Extract_DetInfo(results, detect_class):
    """
    Extract and process detection information from model results.
    :param results: Model predictions including location, class, and confidence of the detected object.
    :param detect_class: Index of the target class to be extracted.
    :return: Detected object locations and confidence levels.
    """
    detections = np.empty((0, 4))
    conf_array = []

    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                conf_array.append(conf)
    return detections, conf_array
