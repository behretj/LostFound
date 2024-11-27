import os
import argparse
import numpy as np
import open3d as o3d
import cv2
from .yolo_integration import register_drawers, register_light_switches
from .camera_transforms import pose_ipad_pointcloud, pose_aria_pointcloud, icp_alignment


def preprocess_scan(
    scan_dir: str,
    drawer_detection: bool = False,
    light_switch_detection: bool = False,
    marker_type: int = cv2.aruco.DICT_APRILTAG_36h11,
    marker_id: int = 52,
    aruco_length: float = 0.148
) -> np.ndarray | None:
    """
    Preprocesses an iPad scan by performing optional object detections and updating 3D mask predictions.

    This function processes the specified scan directory to perform drawer and light switch detections,
    and searches for a specified ArUco marker in the image sequence. Detected drawers are used to update
    the 3D mask prediction file, and if the specified ArUco marker is found, the function returns a 
    4x4 transformation matrix representing the marker's pose.

    :param scan_dir: Path to the directory containing the scan images.
    :param drawer_detection: Flag indicating whether to perform drawer detection.
    :param light_switch_detection: Flag indicating whether to perform light switch detection.
    :param marker_type: ArUco marker dictionary type for marker detection.
    :param marker_id: ID of the specific ArUco marker to detect.
    :param aruco_length: Physical length of the ArUco marker in meters, used for pose estimation.
    :return: 4x4 numpy array representing the pose of the detected marker, or None if no marker is found.
    """
    with open(scan_dir + "/predictions.txt", 'r') as file:
        lines = file.readlines()

    pcd = o3d.io.read_point_cloud(scan_dir + "/mesh_labeled.ply")
    points = np.asarray(pcd.points)

    if drawer_detection and not os.path.exists(scan_dir + "/predictions_drawers.txt"):
        if os.path.exists(scan_dir + "/predictions_light_switches.txt"):
            with open(scan_dir + "/predictions_light_switches.txt", 'r') as file:
                light_lines = file.readlines()
        
            next_line = len(lines) + len(light_lines)
        else:
            next_line = len(lines)
        
        indices_drawers = register_drawers(scan_dir)
        
        drawer_lines=[]
        for indices_drawer in indices_drawers:
            binary_mask = np.zeros(points.shape[0])
            binary_mask[indices_drawer] = 1
            np.savetxt(scan_dir + f"/pred_mask/{next_line:03}.txt", binary_mask, fmt='%d')
            drawer_lines += [f"pred_mask/{next_line:03}.txt 25 1.0\n",]
            next_line += 1
        
        with open(scan_dir + "/predictions_drawers.txt", 'a') as file:
            file.writelines(drawer_lines)
    
    if light_switch_detection and not os.path.exists(scan_dir + "/predictions_light_switches.txt"):
        if os.path.exists(scan_dir + "/predictions_drawers.txt"):
            with open(scan_dir + "/predictions_drawers.txt", 'r') as file:
                drawer_lines = file.readlines()
            
            next_line = len(lines) + len(drawer_lines)
        else:
            next_line = len(lines)

        indices_lights = register_light_switches(scan_dir)
        
        light_lines = []
        for indices_light in indices_lights:
            binary_mask = np.zeros(points.shape[0])
            binary_mask[indices_light] = 1
            np.savetxt(scan_dir + f"/pred_mask/{next_line:03}.txt", binary_mask, fmt='%d')
            light_lines += [f"pred_mask/{next_line:03}.txt 232 1.0\n",]
            next_line += 1
    
        with open(scan_dir + "/predictions_light_switches.txt", 'a') as file:
            file.writelines(light_lines)
    
    if not os.path.exists(scan_dir + "/aruco_pose.npy"):
        T_ipad = pose_ipad_pointcloud(scan_dir, marker_type=marker_type, id=marker_id, aruco_length=aruco_length)
        if T_ipad is None:
            return None
        np.save(scan_dir + "/aruco_pose.npy", T_ipad)
    else:
        T_ipad = np.load(scan_dir + "/aruco_pose.npy")
    
    return T_ipad
    
def preprocess_aria(
    scan_dir: str,
    aria_dir: str,
    marker_type: int = cv2.aruco.DICT_APRILTAG_36h11,
    marker_id: int = 52,
    aruco_length: float = 0.148
) -> np.ndarray | None:
    """
    Preprocesses data from Aria and iPad scans, performing marker detection and pose estimation.

    This function loads data from both the scan directory and the Aria device directory, detecting a 
    specified ArUco marker in the image sequences. If the specified marker is found, it returns a 
    4x4 transformation matrix representing the marker’s pose relative to the camera.

    :param scan_dir: Path to the directory containing the iPad scan images.
    :param aria_dir: Path to the directory containing the Aria device data.
    :param marker_type: ArUco marker dictionary type for marker detection.
    :param marker_id: ID of the specific ArUco marker to detect.
    :param aruco_length: Physical length of the ArUco marker in meters, used for pose estimation.
    :return: 4x4 numpy array representing the pose of the detected marker, or None if no marker is found.
    """
    if not os.path.exists(aria_dir  + "/icp_aligned_pose.npy"):
        if os.path.exists(aria_dir + "/aruco_pose.npy"):
            T_aria = np.load(aria_dir + "/aruco_pose.npy")
        else:
            T_aria = pose_aria_pointcloud(aria_dir, marker_type=marker_type, id=marker_id, aruco_length=aruco_length)
            if T_aria is None:
                return None
            np.save(aria_dir + "/aruco_pose.npy", T_aria)
        
        T_ipad = np.load(scan_dir + "/aruco_pose.npy")

        T_scan_aria = icp_alignment(scan_dir, aria_dir, T_init=np.dot(T_aria, np.linalg.inv(T_ipad)))
        np.save(aria_dir + "/icp_aligned_pose.npy", T_scan_aria)
    
    return True

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Preprocess the iPad Scan.')
   parser.add_argument('--scan_dir', type=str, required=True, help='Path to the "all data" folder from the 3D iPad scan.')
   args = parser.parse_args()
   preprocess_scan(args.scan_dir)