from __future__ import absolute_import, division, print_function
import os
import glob
import pickle
import cv2
import numpy as np
from typing import Generator, Optional

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose, get_nearest_pose
import projectaria_tools.core.mps as mps

import thirdparty.detector._init_paths
from model.utils.net_utils import vis_detections_filtered_objects_PIL
from .detection_utils import hand_object_detection, load_faster_rcnn

from src import crop_image


def data_loader(
    scan_dir: str,
    force_object_detection: bool = False,
    vis_detections: bool = False
) -> Generator[tuple[int, np.ndarray, list, list, Optional[np.ndarray], Optional[np.ndarray], int, np.ndarray], None, None]:
    """
    A generator function that loads and processes data from a VRS file captured by head-mounted Aria glasses.

    This data loader processes a VRS file within the specified directory, yielding data for each observation 
    frame by frame. Each yielded result includes camera pose, detected hands and objects, palm positions, 
    timestamps, and the image itself. Optional parameters allow forced object detection and visualization 
    of detections.

    :param scan_dir: Path to the directory containing the VRS file data.
    :param force_object_detection: Flag to enforce object detection on each frame, even if detections already exist. Defaults to False.
    :param vis_detections: Flag to enable visualization of detections in each frame. Defaults to False.
    :yield: Tuple containing:
        - frame_id: Integer identifier for the current frame.
        - camera_pose: 4x4 numpy array representing the camera’s pose in the world frame.
        - hand_detections: List of detections for hands, if detected in the frame.
        - object_detections: List of detected objects in the frame.
        - left_palm: Position of the left palm in the 3D space, if detected.
        - right_palm: Position of the right palm in the 3D space, if detected.
        - time_stamp: Timestamp of the current frame.
        - image: Numpy array representing the image captured in the frame.
    """
    fasterRCNN = None

    ### Load the necessary files
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"

    camera_label = "camera-rgb"
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    stream_id = provider.get_stream_id_from_label(camera_label)
    w, h = calib.get_image_size()


    calib_device = provider.get_device_calibration()
    T_device_camera = calib_device.get_transform_device_sensor(camera_label).to_matrix()
    
    pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)

    detection_file = f'{scan_dir}/detection_results.pickle'
    if not os.path.exists(detection_file) or force_object_detection:
        print("No detection file found in directory.")

        fasterRCNN = load_faster_rcnn()

        force_object_detection = True
        detection_results = dict()
    else:
        with open(detection_file, "rb") as f:
            detection_results = pickle.load(f)
    
    if vis_detections:
        detection_path = os.path.join(scan_dir, "detections")
        os.makedirs(detection_path, exist_ok=True)
    
    wrist_and_palm_poses_path = scan_dir + "/mps_" + filename + "_vrs/hand_tracking/wrist_and_palm_poses.csv"
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    if len(wrist_and_palm_poses) == 0 or len(closed_loop_traj) == 0:
        print(len(wrist_and_palm_poses), len(closed_loop_traj))
        print("One of the provided files (hand positions, camera poses) is empty.")
        return

    for index in range(provider.get_num_data(stream_id)):
        observation = [index, None, None, None, None, None, None, None]

        name_curr = f"frame_{index:05}.jpg"

        image_data = provider.get_image_data_by_index(stream_id, index)
        query_timestamp = image_data[1].capture_timestamp_ns
        observation[6] = query_timestamp
        raw_image = image_data[0].to_numpy_array()
        undistorted_image = calibration.distort_by_calibration(raw_image, pinhole, calib)
        observation[7] = crop_image(undistorted_image)

        
        name_curr = f"frame_{index:05}.png"
        if force_object_detection:
            img = cv2.rotate(observation[7], cv2.ROTATE_90_CLOCKWISE)
            hand_dets, obj_dets = hand_object_detection(img[..., ::-1], fasterRCNN)
            if hand_dets is None or obj_dets is None:
                img = cv2.flip(img, 1)
                hand_dets, obj_dets = hand_object_detection(img[..., ::-1], fasterRCNN)
            image_results = {"hand_dets": hand_dets, "obj_dets": obj_dets}
            detection_results[name_curr] = image_results
        
        image_info = detection_results[name_curr]
        hand_dets, obj_dets = image_info["hand_dets"], image_info["obj_dets"]
        if vis_detections:
            im2show = vis_detections_filtered_objects_PIL(cv2.rotate(observation[7][..., ::-1], cv2.ROTATE_90_CLOCKWISE), hand_dets, obj_dets, 0.5, 0.5)
            result_path = os.path.join(detection_path, name_curr)
            im2show.save(result_path)
        
        observation[2] = hand_dets
        observation[3] = obj_dets

        device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)

        if device_pose is None:
            yield observation
            continue

        T_world_device = device_pose.transform_world_device.to_matrix()
        observation[1] = np.dot(T_world_device, T_device_camera)

        wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, query_timestamp)

        if wrist_and_palm_pose is None:
            yield observation
            continue
        
        
        # Time difference between query timestamp and found pose timestamps is too large (> 0.1s)
        if abs(device_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8 or \
            abs(wrist_and_palm_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8:
            yield observation
            continue

        if wrist_and_palm_pose.left_hand.confidence > 0.0:
            observation[4] = np.dot(T_world_device, np.append(wrist_and_palm_pose.left_hand.palm_position_device, 1))[:3]
        
        if wrist_and_palm_pose.right_hand.confidence > 0.0:
            observation[5] = np.dot(T_world_device, np.append(wrist_and_palm_pose.right_hand.palm_position_device, 1))[:3]
        
        yield observation        

    with open(detection_file, 'wb') as f:
        pickle.dump(detection_results, f)
    
    return