import cv2
import numpy as np
import open3d as o3d
import os, json, glob
from projectaria_tools.core.mps.utils import get_nearest_pose
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import filter_points_from_confidence

def pose_aria_pointcloud(
    scan_dir: str,
    marker_type: int = cv2.aruco.DICT_APRILTAG_36h11,
    id: int = 52,
    aruco_length: float = 0.148,
    save_aria_pcd: bool = True,
    vis_detection: bool = False,
    vis_poses: bool = False
) -> np.ndarray | None:
    """
    Finds and returns the pose of the first ArUco marker in the world frame within a VRS file.

    This function scans the specified directory for a VRS file, identifies the first occurrence of the specified 
    ArUco marker, and calculates its pose in the world frame. If no marker is found, a warning is printed, and 
    the function returns None. The pose can be visualized, and the resulting point cloud can optionally be saved.

    :param scan_dir: Path to the directory containing the VRS file with the scan data.
    :param marker_type: Type of ArUco marker dictionary used for detection. Defaults to AprilTag 36h11.
    :param id: ID of the specific ArUco marker to detect. Defaults to 52.
    :param aruco_length: Physical length of the ArUco marker in meters, used for pose estimation. Defaults to 0.148.
    :param save_aria_pcd: Flag indicating whether to save the generated point cloud of the Aria data. Defaults to True.
    :param vis_detection: Flag to enable visualization of the detection process. Defaults to False.
    :param vis_poses: Flag to enable visualization of the calculated poses. Defaults to False.
    :return: 4x4 numpy array representing the pose of the detected marker in the world frame, or None if no marker is found.
    """
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"

    # Point cloud creation
    global_points_path = scan_dir + "/mps_" + filename + "_vrs/slam/semidense_points.csv.gz"
    points = mps.read_global_point_cloud(global_points_path)

    # filter the point cloud using thresholds on the  inverse depth and distance standard deviation, user set
    inverse_distance_std_threshold = 0.005
    distance_std_threshold = 0.001

    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

    pcd = o3d.geometry.PointCloud()
    points = np.array([point.position_world for point in filtered_points])

    pcd.points = o3d.utility.Vector3dVector(points)
    # paint point cloud in random color
    pcd.paint_uniform_color(np.random.rand(3))

    
    # world poses
    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    valid_poses = [closed_loop_traj[0].tracking_timestamp.total_seconds()*1e9, closed_loop_traj[-1].tracking_timestamp.total_seconds()*1e9]
    
    camera_label = "camera-rgb"
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    stream_id = provider.get_stream_id_from_label(camera_label)

    w, h = calib.get_image_size()
    pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

    cam_matrix = np.array([[calib.get_focal_lengths()[0], 0, calib.get_principal_point()[0]],
                        [0, calib.get_focal_lengths()[1], calib.get_principal_point()[1]],
                        [0, 0, 1]])
    
    arucoDict = cv2.aruco.getPredefinedDictionary(marker_type)
    arucoParams = cv2.aruco.DetectorParameters()

    for i in range(provider.get_num_data(stream_id)):
        image_data = provider.get_image_data_by_index(stream_id, i)
        # Ensure that we have a valid pose for the image
        if image_data[1].capture_timestamp_ns < valid_poses[0] or image_data[1].capture_timestamp_ns > valid_poses[1]:
            continue
        raw_image = image_data[0].to_numpy_array()
        undistorted_image = calibration.distort_by_calibration(raw_image, pinhole, calib)
        aruco_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
        aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_CLOCKWISE)
        
        corners, ids, _ = cv2.aruco.detectMarkers(aruco_image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            matching = np.array(ids)==id
            if not np.any(matching): continue
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_length, cam_matrix, 0)
            rotation_3x3, _ = cv2.Rodrigues(rvecs)
            T_camera_marker = np.eye(4)
            T_camera_marker[:3, :3] = rotation_3x3
            T_camera_marker[:3, 3] = tvecs

            # for debugging: visualize the aruco detection
            if vis_detection:
                cv2.aruco.drawDetectedMarkers(aruco_image, corners, ids)
                cv2.drawFrameAxes(aruco_image, cam_matrix, 0, rvecs, tvecs, 0.1)
                scale = 0.4
                dim = (int(aruco_image.shape[1] * scale), int(aruco_image.shape[0] * scale))
                resized = cv2.resize(aruco_image, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow("aruco", resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            pose_info = get_nearest_pose(closed_loop_traj, image_data[1].capture_timestamp_ns)
            assert pose_info, "could not find pose for timestamp"
            T_world_device = pose_info.transform_world_device
            T_device_camera = calib.get_transform_device_camera()
            T_world_camera = T_world_device @ T_device_camera
            T_world_camera = T_world_camera.to_matrix()
            
            rot_z_270 = np.array([[np.cos(3 * np.pi / 2), -np.sin(3 * np.pi / 2), 0, 0],
                                [np.sin(3 * np.pi / 2), np.cos(3 * np.pi / 2), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

            
            T_world_camera = np.dot(T_world_camera, rot_z_270)

            T_world_marker = np.dot(T_world_camera, T_camera_marker)

            # for debugging: visualize the poses
            if vis_poses:
                mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera.transform(T_world_camera)
                mesh_frame_marker.transform(T_world_marker)

                world_origin = np.array([0, 0, 0, 1])
                camera_origin = np.dot(T_world_camera, world_origin)[:3]
                marker_origin = np.dot(T_world_marker, world_origin)[:3]


                sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_marker.paint_uniform_color([1, 0, 0]) # red
                sphere_marker.translate(marker_origin)

                sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_camera.paint_uniform_color([0, 1, 0]) # green
                sphere_camera.translate(camera_origin)

                o3d.visualization.draw_geometries([pcd, mesh_frame_world, mesh_frame_camera, mesh_frame_marker, sphere_camera, sphere_marker])
            
            if save_aria_pcd:
                pcd_path = os.path.join(scan_dir, "aria_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)

            return T_world_marker
    
    print("No marker found for pose estimation")

def pose_ipad_pointcloud(
    scan_dir: str,
    pcd_path: str = None,
    marker_type: int = cv2.aruco.DICT_APRILTAG_36h11,
    id: int = 52,
    aruco_length: float = 0.148,
    vis_detection: bool = False
) -> np.ndarray | None:
    """
    Finds and returns the pose of the first ArUco marker in the world frame within an iPad scan.

    This function scans the specified directory for an iPad scan file, identifies the first occurrence of the specified 
    ArUco marker, and calculates its pose in the world frame. If no marker is found, the function prints a warning 
    and returns None. Optionally, a point cloud file path can be provided, and detection visualization can be enabled.

    :param scan_dir: Path to the directory containing the iPad scan data.
    :param pcd_path: Optional path to save or load the point cloud data associated with the iPad scan. Defaults to None.
    :param marker_type: Type of ArUco marker dictionary used for detection. Defaults to AprilTag 36h11.
    :param id: ID of the specific ArUco marker to detect. Defaults to 52.
    :param aruco_length: Physical length of the ArUco marker in meters, used for pose estimation. Defaults to 0.148.
    :param vis_detection: Flag to enable visualization of the detection process. Defaults to False.
    :return: 4x4 numpy array representing the pose of the detected marker in the world frame, or None if no marker is found.
    """
    image_files = sorted(glob.glob(os.path.join(scan_dir, 'frame_*.jpg')))

    for image_name in image_files:
        image = cv2.imread(image_name)
        ### For the first iPad scan, the image needs to be rotated 90 degree        
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        with open(image_name[:-4] + ".json", 'r') as f:
            camera_info = json.load(f)


        cam_matrix = np.array(camera_info["intrinsics"]).reshape(3, 3)
        
        arucoDict = cv2.aruco.getPredefinedDictionary(marker_type)
        arucoParams = cv2.aruco.DetectorParameters()


        corners, ids, _ = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            matching = np.array(ids)==id
            if not np.any(matching): continue
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_length, cam_matrix, 0)
            rotation_3x3, _ = cv2.Rodrigues(rvecs)
            T_camera_marker = np.eye(4)
            T_camera_marker[:3, :3] = rotation_3x3
            T_camera_marker[:3, 3] = tvecs

            # for debugging: visualize the aruco detection
            if vis_detection:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.drawFrameAxes(image, cam_matrix, 0, rvecs, tvecs, 0.1)
                scale = 0.4
                dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow("ipad", resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            T_world_camera = np.array(camera_info["cameraPoseARFrame"]).reshape(4, 4)
            
            rot_x_180 = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
            
            
            ### For the first iPad scan, the camera needs to be rotated 180 degree
            ### around y instead of x axis (and the rotation of the image in the beginning)
            # T_world_camera = np.dot(T_world_camera, rot_y_180)
            
            ### For the second iPad scan:
            T_world_camera = np.dot(T_world_camera, rot_x_180)

            T_world_marker = np.dot(T_world_camera, T_camera_marker)

            if pcd_path is not None:
                pcd = o3d.io.read_point_cloud(pcd_path)

                mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                mesh_frame_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                
                mesh_frame_camera.transform(T_world_camera)
                mesh_frame_marker.transform(T_world_marker)

                world_origin = np.array([0, 0, 0, 1])
                camera_origin = np.dot(T_world_camera, world_origin)[:3]
                marker_origin = np.dot(T_world_marker, world_origin)[:3]


                sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_marker.paint_uniform_color([1, 0, 0]) # red
                sphere_marker.translate(marker_origin)

                sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                sphere_camera.paint_uniform_color([0, 1, 0]) # green
                sphere_camera.translate(camera_origin)

                o3d.visualization.draw_geometries([pcd, mesh_frame_world, mesh_frame_camera, mesh_frame_marker, sphere_camera, sphere_marker])
            
            return T_world_marker
        
    print("No marker found for pose estimation")

def icp_alignment(
    source_folder: str,
    target_folder: str,
    T_init: np.ndarray = np.eye(4)
) -> np.ndarray:
    """
    Aligns the source point cloud to the target point cloud using the Iterative Closest Point (ICP) algorithm.

    This function loads point cloud data from the specified source and target folders, then aligns the source 
    point cloud to the target point cloud by applying the ICP algorithm. An initial transformation matrix can 
    be provided to guide the alignment process. The function returns the transformation matrix that aligns 
    the source to the target.

    :param source_folder: Path to the folder containing the source point cloud data.
    :param target_folder: Path to the folder containing the target point cloud data.
    :param T_init: Initial 4x4 transformation matrix to start the alignment. Defaults to the identity matrix.
    :return: 4x4 numpy array representing the transformation matrix aligning the source to the target.
    """
    source_pcd = o3d.io.read_point_cloud(source_folder + "/mesh_labeled.ply")
    target_pcd = o3d.io.read_point_cloud(target_folder + "/aria_pointcloud.ply")

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, 0.05, T_init,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))
    
    source_pcd.transform(reg_p2l.transformation)
    
    return np.array(reg_p2l.transformation)
