import open3d as o3d
import numpy as np
import cv2, os, glob, pickle
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from math import ceil
from .drawer_detection import predict_yolodrawer
from .light_switch_detection import predict_light_switches
import scipy.cluster.hierarchy as hcluster
from src import parse_json, parse_txt
from .projecting import project_points_bbox, detections_to_bboxes
from collections import namedtuple

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["file", "name", "conf", "bbox"])

def compute_iou(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) between two arrays.

    :param array1: First array.
    :param array2: Second array.
    :return: IoU score as a float, representing the overlap between the two arrays.
    """
    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)
    iou = len(intersection) / len(union)
    return iou

def dynamic_threshold(detection_counts: list, n_clusters: int = 2) -> float:
    """
    Calculates a dynamic threshold for detection count differences using k-means clustering.

    This function computes the differences between consecutive detection counts, clusters the differences 
    using k-means, and calculates a threshold based on the cluster centers. The threshold is set to the midpoint 
    between the two closest cluster centers, providing a dynamic way to separate high and low change rates in detection counts.

    :param detection_counts: List of detection counts over a sequence, used to calculate consecutive differences.
    :param n_clusters: Number of clusters for k-means. Defaults to 2.
    :return: Calculated threshold as a float, representing the midpoint between cluster centers.
    """
    differences = np.array([abs(j - i) for i, j in zip(detection_counts[:-1], detection_counts[1:])]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(differences)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    
    if len(cluster_centers) > 1:
        threshold = (cluster_centers[0] + cluster_centers[1]) / 2
    else:
        threshold = cluster_centers[0]
    
    return threshold

def cluster_detections(
    detections: list,
    points_3d: np.ndarray,
    aligned: bool = False
) -> tuple[int, str, str, list[np.ndarray]]:
    """
    Clusters 3D detection points and organizes results with metadata.

    This function clusters detected points in 3D space based on the input detections and their coordinates.
    It optionally aligns the detections if the `aligned` flag is set. The function returns metadata about the 
    clustered detections and a list of bounding boxes in 3D space for each cluster.

    :param detections: List of detection data, containing details for each detected object or region.
    :param points_3d: Array of 3D points corresponding to the detections, typically of shape (N, 3).
    :param aligned: Flag indicating if detections should be aligned prior to clustering. Defaults to False.
    :return: Tuple containing:
        - data_num: Integer representing the number of clusters or detections processed.
        - data_name: String indicating the name or identifier for the detection dataset.
        - data_file: String representing the filename or source of the detection data.
        - points_bb_3d_list: List of 3D numpy arrays, each representing the bounding box points for a cluster.
    """
    if not detections:
        return []
    dels = []
    for idx, det in enumerate(detections):
        if det[1] == 0:
            dels.append(idx)

    detections_filtered = [item for i, item in enumerate(detections) if i not in dels]

    data_file = []
    data_name = []
    data_num = []
    for dets in detections_filtered:
        dets_per_image = dets[0]
        for det in dets_per_image:
            # data.append([det.file, det.conf, det.name, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])
            data_name.append(det.name)
            data_num.append([det.conf, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])
            data_file.append(det.file)

    data_num = np.array(data_num)
    data_name = np.array(data_name)
    data_file = np.array(data_file)

    center_coord_3d = []
    center_index = []
    points_bb_3d_list = []
    for idx, det in enumerate(data_num):
        bbox = det[1:5]

        if aligned:
            intrinsics, _ = parse_json(data_file[idx]+ ".json")
            cam_pose = parse_txt(data_file[idx]+ ".txt")
        else:
            intrinsics, cam_pose = parse_json(data_file[idx]+ ".json")

        image = cv2.imread(data_file[idx] + ".jpg")
        width, height = image.shape[1], image.shape[0]

        _, points_bb_3d = project_points_bbox(points_3d, cam_pose, intrinsics, width, height, bbox.copy())

        centroid = np.mean(points_bb_3d, axis=0)
        dist = np.linalg.norm(points_3d - centroid, axis=1)
        closest_index = np.argmin(dist)
        closest_point = points_3d[closest_index]

        center_coord_3d.append(closest_point)
        center_index.append(closest_index)
        points_bb_3d_list.append(points_bb_3d)

    center_coord_3d = np.array(center_coord_3d)
    center_index = np.array(center_index)

    clusters = hcluster.fclusterdata(center_coord_3d, 0.15, criterion="distance")
    data_num = np.column_stack((data_num, center_coord_3d, center_index, clusters))
    return data_num, data_name, data_file, points_bb_3d_list

def cluster_images(detections: list) -> list:
    """
    Groups temporally close images based on detection data into clusters.

    :param detections: List of the detection entries and the corresponding number of detections for each image, used for clustering.
    :return: List of clusters.
    """
    if not detections:
        return []
    
    detection_counts = [n for (_, n) in detections]
    
    threshold = ceil(dynamic_threshold(detection_counts))
    clusters = []
    current_cluster = []

    for index, count in enumerate(detection_counts):
        if not current_cluster or (index > 0 and abs(detection_counts[index - 1] - count) <= threshold):
            current_cluster.append((index, count))
        else:
            if current_cluster[-1][1] > 0: 
                clusters.append(current_cluster)
            current_cluster = [(index, count)]
    
    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def select_optimal_images(clusters: list) -> list:
    """
    Selects the optimal image from each cluster based on a scoring criterion - the imae
    with the maximum number of detections in the cluster.

    :param clusters: List of clusters, where each cluster contains tuples of images and their scores.
    :return: List of optimal images, with one image selected per cluster based on the highest score.
    """
    optimal_images = []
    for cluster in clusters:
        if cluster:
            optimal_images.append(max(cluster, key=lambda x: x[1])[0])
    return optimal_images

def register_drawers(dir_path: str) -> list:
    """
    Registers drawers from a YOLO detection algorithm in the 3D scene.

    :param dir_path: Path to the directory containing drawer data for registration.
    :return: List of sorted indices representing registered drawers.
    """
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections.pkl')):
        with open(os.path.join(dir_path, 'detections.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            # img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_yolodrawer(image, image_name[:-4], vis_block=False)]
        with open(os.path.join(dir_path, 'detections.pkl'), 'wb') as f:
            pickle.dump(detections, f)
        
    clusters = cluster_images(detections)
    
    optimal_images = select_optimal_images(clusters)
    
    detections = [det for subdets in [detections[opt][0] for opt in optimal_images] for det in subdets]
    
    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labeled.ply'))
    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections, threshold=0.9)

    all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(pcd_original.points)), conf) for bbox, conf in bboxes_3d]

    registered_indices = []
    for indcs, conf in all_bbox_indices:     
        for idx, (reg_indcs, confidence) in enumerate(registered_indices):
            iou = compute_iou(reg_indcs, indcs)
            if iou > 0.1:  # Check if the overlap is greater than 10%
                if conf > confidence:
                    registered_indices[idx] = (indcs, conf)
                break
        else:
            registered_indices.append((indcs, conf))
    
    return [indcs for (indcs, _) in sorted(registered_indices, key=lambda x: x[1])]


def register_light_switches(dir_path: str, vis_block: bool = False) -> list:
    """
    Registers light switches from a YOLO detection algorithm in the 3D scene.

    This function processes data within a specified directory to identify and register light switches detected by YOLO 
    in a 3D scene. Optionally, visualization can be turned on during the registration process.

    :param dir_path: Path to the directory containing light switch data for registration.
    :param vis_block: Flag indicating whether to have visualization during processing. Defaults to False.
    :return: List of sorted indices representing registered light switches.
    """
    # stores tuples containing the detected box(es) and its/their confidence(s)
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections_lightswitch.pkl')):
        with open(os.path.join(dir_path, 'detections_lightswitch.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            # img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_light_switches(image, image_name[:-4])]
        with open(os.path.join(dir_path, 'detections_lightswitch.pkl'), 'wb') as f:
            pickle.dump(detections, f)

    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labeled.ply'))
    points = np.asarray(pcd_original.points)

    data_num, data_name, data_file, points_bb_3d_list = cluster_detections(detections, points)
    num_clusters = len(np.unique(data_num[:, -1]))
    detections = []
    test_centroids_idx = []
    for cluster in range(1, num_clusters+1):
        idx = np.where(data_num[:, -1] == cluster)
        idx_start = np.min(idx)
        det_per_cluster = data_num[data_num[:, -1] == cluster]

        optimal_detection_idx = np.argmax(det_per_cluster[:, 0]) + idx_start

        file = data_file[optimal_detection_idx]
        name = data_name[optimal_detection_idx]
        bbox = BBox(xmin=data_num[optimal_detection_idx][1], ymin=data_num[optimal_detection_idx][2],
                    xmax=data_num[optimal_detection_idx][3], ymax=data_num[optimal_detection_idx][4])
        detections.append(Detection(file=file, name=name, conf=data_num[optimal_detection_idx][0], bbox=bbox))
        test_centroids_idx.append(data_num[optimal_detection_idx][-2])

    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections, threshold=0.9)

    all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(pcd_original.points)), conf) for bbox, conf in bboxes_3d]

    registered_indices = []
    for indcs, conf in all_bbox_indices:
        for idx, (reg_indcs, confidence) in enumerate(registered_indices):
            iou = compute_iou(reg_indcs, indcs)
            if iou > 0.1:  # Check if the overlap is greater than 10%
                if conf > confidence:
                    registered_indices[idx] = (indcs, conf)
                break
        else:
            registered_indices.append((indcs, conf))


    if vis_block:
        # highlight bboxes
        all_colors = np.asarray(pcd_original.colors)
        for (ind, conf) in all_bbox_indices:
            all_colors[ind] = np.random.rand(3)
        pcd_original.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.visualization.draw_geometries([pcd_original])

    return [indcs for (indcs, _) in sorted(registered_indices, key=lambda x: x[1])]

if __name__ == "__main__":
    pass
