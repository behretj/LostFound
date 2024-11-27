import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src import parse_json

def project_points_bbox(
    points_3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    bbox: tuple,
    grid: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points onto a 2D plane within a specified bounding box, using camera parameters.

    This function takes a set of 3D points and projects them onto a 2D image plane, applying specified
    extrinsic and intrinsic camera parameters. The projection is limited to the defined bounding box
    and uses a grid to sample points within the bounding box.

    :param points_3d: Array of 3D points to be projected, typically of shape (N, 3).
    :param extrinsics: 4x4 transformation matrix representing the camera's extrinsic parameters.
    :param intrinsics: 3x3 matrix representing the camera's intrinsic parameters.
    :param width: Width of the target 2D projection plane in pixels.
    :param height: Height of the target 2D projection plane in pixels.
    :param bbox: Tuple defining the bounding box (xmin, ymin, xmax, ymax) within which to limit the projection.
    :param grid: Grid size for sampling points within the bounding box to compensate for reduced number of 3D points in the given point cloud. Defaults to 15.
    :return: Tuple containing:
        - valid_image_points: 2D numpy array of points within the bounding box on the 2D plane.
        - valid_points_3d: 3D numpy array of the corresponding 3D points for each valid 2D point.
    """
    extrinsics = np.linalg.inv(extrinsics)
    
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    extrinsics =  np.zeros((3,4))
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    points_cam = extrinsics @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    points_cam = points_cam.T
    
    points = intrinsics @ extrinsics @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T

    image_points = points[:2, :] / points[2, :]
    image_points = image_points.T

    
    depth_buffer = np.full((height//grid, width//grid), -np.inf)
    best_points = np.zeros((height//grid, width//grid, 3))
    best_cam_points = np.zeros((height//grid, width//grid, 3))
    
    # correct x-coordinate of bbox:
    bbox[0], bbox[2] = width - bbox[2], width - bbox[0]
    bbox = bbox // grid  
    
    for point, img_pt, cam_pt in zip(points_3d, image_points, points_cam):
        x, y = int(img_pt[0]//grid), int(img_pt[1]//grid)
        if int(bbox[0]) <= x < int(bbox[2]) and int(bbox[1]) <= y < bbox[3]:
            if cam_pt[2] > depth_buffer[y, x]:  # Since z is negative, a smaller value means closer
                depth_buffer[y, x] = cam_pt[2]
                best_points[y, x] = point
                best_cam_points[y, x] = cam_pt
    
    # Filter valid points and their 2D projections
    valid = (depth_buffer != -np.inf)
    valid_points_3d = best_points[valid]
    y_indices, x_indices = np.where(valid)
    valid_image_points = np.vstack((x_indices*grid, y_indices*grid)).T
    
    return valid_image_points, valid_points_3d

def detections_to_bboxes(
    points: np.ndarray,
    detections: list,
    threshold: float = 0.7
) -> list:
    """
    Lifts 2D detection bounding boxes to 3D bounding boxes with confidence scores.

    :param points: Array of 3D points associated with the captured scene of shape (N, 3).
    :param detections: List of 2D detections.
    :param threshold: Confidence threshold for considering a detection valid. Defaults to 0.7.
    :return: List of tuples containing 3D bounding boxes for detections and their associated confidence scores.
    """
    bboxes_3d = []
    for file, _, confidence, bbox in detections:
        intrinsics, extrinsics = parse_json(file + ".json")
        image = cv2.imread(file + ".jpg")
        width, height = image.shape[1], image.shape[0]

        if confidence > threshold:
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            _, points_3d = project_points_bbox(points, extrinsics, intrinsics, width, height, bbox)
            if points_3d.shape[0] < 15:
                continue
            pcd_bbox = o3d.geometry.PointCloud()
            pcd_bbox.points = o3d.utility.Vector3dVector(points_3d)
            _, inliers = pcd_bbox.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            pcd_bbox = pcd_bbox.select_by_index(inliers) 
            bbox_3d = pcd_bbox.get_minimal_oriented_bounding_box()
            bboxes_3d += [(bbox_3d, confidence)]

       
    return bboxes_3d
