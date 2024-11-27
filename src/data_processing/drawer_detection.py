from __future__ import annotations

import os.path
from logging import Logger
from typing import Optional
import shutil

import numpy as np

import cv2
from matplotlib import pyplot as plt
import colorsys
from .docker_communication import save_files, send_request
from collections import namedtuple

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["file", "name", "conf", "bbox"])

COLORS = {
    "door": (0.651, 0.243, 0.957),
    "handle": (0.522, 0.596, 0.561),
    "cabinet door": (0.549, 0.047, 0.169),
    "refrigerator door": (0.082, 0.475, 0.627),
}

CATEGORIES = {"0": "door", "1": "handle", "2": "cabinet door", "3": "refrigerator door"}

def generate_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of visually distinct RGB colors.

    :param n: The number of distinct colors to generate.
    :return: List of RGB color tuples, each containing three floats representing red, green, and blue values.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        lightness = 0.5
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))

    return colors

def draw_boxes(image: np.ndarray, detections: list[Detection], output_path: str) -> None:
    """
    Draws bounding boxes on an image based on detection data and saves the result.

    :param image: Input image as a numpy array on which to draw the bounding boxes.
    :param detections: List of Detection objects, each containing bounding box coordinates and relevant metadata.
    :param output_path: File path to save the output image with drawn bounding boxes.
    :return: None. The function saves the output image to the specified path.
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    names = sorted(list(set([det.name for det in detections])))
    names_dict = {name: i for i, name in enumerate(names)}
    colors = generate_distinct_colors(len(names_dict))

    for _, name, conf, (xmin, ymin, xmax, ymax) in detections:
        w, h = xmax - xmin, ymax - ymin
        color = colors[names_dict[name]]
        ax.add_patch(
            plt.Rectangle((xmin, ymin), w, h, fill=False, color=color, linewidth=6)
        )
        text = f"{name}: {conf:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.savefig(output_path)

def predict_yolodrawer(
    image: np.ndarray,
    image_name: str,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    port: int = 5004,
    input_format: str = "rgb",
    vis_block: bool = False,
) -> tuple[list[Detection], int] | None:
    """
    Runs YOLO-based drawer detection on an input image and returns detected objects.

    This function performs object detection using a YOLO model to identify drawers in the input image.
    Detected objects are returned as a list of `Detection` objects. Optionally, the function can log 
    events, block visualization, and adjust processing time via a timeout.

    :param image: Input image as a numpy array, formatted according to `input_format`.
    :param image_name: Name or identifier for the input image, used for logging or tracking.
    :param logger: Optional logger for recording detection events and statuses.
    :param timeout: Maximum processing time in seconds before detection times out. Defaults to 90 seconds.
    :param port: Port, where the yolo detection algorithm runs. Defaults to port 5004.
    :param input_format: Format of the input image ("rgb" or "bgr"). Defaults to "rgb".
    :param vis_block: Flag indicating whether to block visualization during processing. Defaults to False.
    :return: Tuple containing:
        - detections: List of `Detection` objects representing detected drawers.
        - detection_count: Integer representing the number of detected objects.
        Returns None if detection fails or times out.
    """
    assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    address_details = {'ip': "127.0.0.1", 'port': port, 'route': "yolodrawer/predict"}
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    
    os.makedirs("tmp", exist_ok=True)

    image_prefix = os.path.basename(image_name)
    save_data = [(f"{os.path.splitext(image_prefix)[0]}.npy", np.save, image)]
    image_path, *_ = save_files(save_data, "tmp")

    paths_dict = {"image": image_path}
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, {}, timeout, "tmp")
    if logger:
        logger.info("Received response!")

    # no detections
    if len(contents) == 0:
        if vis_block:
            draw_boxes(image, [], image_name + "_detections.png")
        return [], 0

    classes = contents["classes"]
    confidences = contents["confidences"]
    bboxes = contents["bboxes"]

    detections = []
    for cls, conf, bbox in zip(classes, confidences, bboxes):
        name = CATEGORIES[str(int(cls))]
        if name != "handle":
            det = Detection(image_name, name, conf, BBox(*bbox))
            detections.append(det)

    if vis_block:
        draw_boxes(image, detections, image_name + "_detections.png")
    
    shutil.rmtree("tmp")
    
    return detections, len(detections)
