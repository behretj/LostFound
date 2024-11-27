import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["file", "name", "conf", "bbox"])

def filter_detections_ultralytics(detections, filter_squaredness=True, filter_area=True, filter_within=True):

    detections = detections[0].cpu()
    xyxy = detections.boxes.xyxy.numpy()
    conf = np.atleast_2d(detections.boxes.conf.numpy()).T

    # filter squaredness outliers
    if filter_squaredness:
        squaredness = (np.minimum(xyxy[:, 2] - xyxy[:, 0],
                                  xyxy[:, 3] - xyxy[:, 1]) /
                       np.maximum(xyxy[:, 2] - xyxy[:, 0],
                                  xyxy[:, 3] - xyxy[:, 1]))

        keep_1 = squaredness > 0.5
        xyxy = xyxy[keep_1, :]
        conf = conf[keep_1, :]

    #filter area outliers
    if filter_area:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep_2 = areas < 3*np.median(areas)
        xyxy = xyxy[keep_2, :]
        conf = conf[keep_2, :]

    # filter bounding boxes within larger ones
    if filter_within:
        centers = np.array([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]).T
        keep_3 = np.ones(xyxy.shape[0], dtype=bool)
        x_in_box = (xyxy[:, 0:1] <= centers[:, 0]) & (centers[:, 0] <= xyxy[:, 2:3])
        y_in_box = (xyxy[:, 1:2] <= centers[:, 1]) & (centers[:, 1] <= xyxy[:, 3:4])
        centers_in_boxes = x_in_box & y_in_box
        np.fill_diagonal(centers_in_boxes, False)
        pairs = np.argwhere(centers_in_boxes)
        idx_remove = pairs[np.where(areas[pairs[:, 0]] - areas[pairs[:, 1]] < 0), 0].flatten()
        keep_3[idx_remove] = False
        xyxy = xyxy[keep_3, :]
        conf = conf[keep_3, :]

    bbox = xyxy
    return bbox, conf


def predict_light_switches(image: np.ndarray, image_name: str, vis_block: bool = False):
    model = YOLO('weights/best.pt')
    results_predict = model.predict(source=image, imgsz=1280, conf=0.15, iou=0.4, max_det=9, agnostic_nms=True,
                                    save=False)

    boxes, conf = filter_detections_ultralytics(detections=results_predict)

    if boxes.shape[0] == 0:
        return [], 0

    if vis_block:
        canv = image.copy()
        for box in boxes:
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])

            cv2.rectangle(canv, (xA, yA), (xB, yB), (0, 255, 0), 2)

        plt.imshow(canv)
        plt.show()

    detections = []
    for idx, box in enumerate(boxes):
        det = Detection(file=image_name, name="light switch", conf=conf[idx][0], bbox=BBox(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3]))
        detections.append(det)

    return detections, len(detections)

if __name__ == "__main__":
    pass