from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import open3d as o3d
import os
import cv2
import torch

from src import SceneGraph, data_loader, project_pcd_to_image

from typing import Optional
from collections import deque
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if DEFAULT_DEVICE == "cuda":
    from cotracker.predictor import CoTrackerOnlinePredictor
    from cotracker.utils.visualizer import Visualizer


def check_hand_object_tracker(hand_dets: np.ndarray, object_dets: np.ndarray) -> bool:
    """
    Checks if there are valid hand and object detections.

    :param hand_dets: Array of hand detections with confidence scores.
    :param object_dets: Array of object detections with confidence scores.
    :return: True if valid detections are present, False otherwise.
    """
    return (hand_dets is not None) and (object_dets is not None) \
        and (any(hand_dets[i, 4] > 0.5 for i in range(hand_dets.shape[0]))) \
        and (any(object_dets[i, 4] > 0.5 for i in range(object_dets.shape[0])))


def hand_velocity(hand_observations: list) -> float:
    """
    Computes the average velocity of hand movements.

    :param hand_observations: List of hand position observations.
    :return: Average velocity of the hand.
    """
    displacements = [np.linalg.norm(hand_observations[i+1] - hand_observations[i]) for i in range(len(hand_observations)-1) if (hand_observations[i+1] is not None) and (hand_observations[i] is not None)]
    if len(hand_observations) < 2 or len(displacements) == 0:
        return 0
    return sum(displacements) / len(displacements)

def _process_step(
    cotracker: CoTrackerOnlinePredictor,
    window_frames: list,
    is_first_step: bool,
    queries: torch.Tensor
) -> tuple:
    """
    Processes a step in the CoTracker sequence.

    This function prepares a video chunk from the window frames and processes it
    using the CoTracker model to update tracking predictions.

    :param cotracker: The CoTracker model for online prediction.
    :param window_frames: List of frames in the current window.
    :param is_first_step: Flag indicating if this is the first step in the sequence.
    :param queries: Tensor of 2D query points for tracking.
    :return: Predicted tracks and visibility from the CoTracker.
    """
    video_chunk = (
        torch.tensor(np.stack(window_frames[-cotracker.step * 2 :]), device=DEFAULT_DEVICE)
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return cotracker(
        video_chunk,
        is_first_step=is_first_step,
        queries=queries,
    )

def compute_rotation(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    queries_3D: np.ndarray,
    ind: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the rotation and translation of an object in the camera frame.

    This function uses the predicted tracks and visibility to solve the PnP problem,
    estimating the rotation and translation vectors of the object.

    :param pred_tracks: Predicted 2D tracks of the object.
    :param pred_visibility: Visibility mask for the predicted tracks.
    :param queries_3D: 3D query points corresponding to the 2D tracks.
    :param ind: Index of the current frame in the sequence.
    :param camera_matrix: Intrinsic camera matrix.
    :param dist_coeffs: Distortion coefficients for the camera.
    :return: Rotation matrix and translation vector of the object.
    """
    current_tracks = pred_tracks[:, ind, :, :].squeeze(0).cpu().numpy().astype(np.float32)
    current_visibility = pred_visibility[:, ind, :].squeeze(0).cpu().numpy()

    success, rvec_obj_cam, tvec_obj_cam, inliers = cv2.solvePnPRansac(queries_3D[current_visibility], current_tracks[current_visibility], camera_matrix, dist_coeffs)
    if not success:
        raise ValueError("PnP failed to find a solution.")            
    R_obj_cam, _ = cv2.Rodrigues(rvec_obj_cam)

    return R_obj_cam, tvec_obj_cam.flatten()


def track(
    scene_graph: SceneGraph,
    scan_dir: str,
    video_path: Optional[str] = None,
    force_object_detection: bool = False,
    save_pose: bool = False,
    tracker_offset: Optional[np.ndarray] = None,
    use_hand: bool = True
) -> None:
    """
    Executes the tracking process for objects within the scene.

    This method iterates over incoming observations from the `aria_data` iterator,
    processes each observation to update object poses based on hand detections and
    object detections, and manages the rendering and saving of tracking data.

    The tracking process involves:
        - Updating the context window with new observations.
        - Checking for hand-object detections and updating object associations.
        - Processing left and right hands to manage tracked objects.
        - Saving poses and timestamps if enabled.
        - Rendering tracking states to images if video rendering is enabled.

    :param scene_graph: The scene graph containing object and spatial information.
    :param scan_dir: Directory containing scan data and where results will be saved.
    :param video_path: Path where tracking visualization video will be saved.
    :param force_object_detection: Whether to force object detection in each frame.
    :param save_pose: Whether to save tracked poses to disk.
    :param tracker_offset: Optional offset to apply to tracked poses.
    :param use_hand: Whether to use hand positions for pose estimation.
    :return: None
    """
    if video_path:
        images = []
        size = (1296, 1296)
        intrinsics = np.array([
            [611.428, 0, 703.5],
            [0, 611.428, 703.5],
            [0, 0, 1]
        ])

        intrinsics, extrinsics = scene_graph.set_camera(size, intrinsics)
        render = o3d.visualization.rendering.OffscreenRenderer(size[0], size[1])
        geometries = scene_graph.scene_geometries()
        for geometry, name, material in geometries:
            render.scene.add_geometry(name, geometry, material)
        render.setup_camera(intrinsics, extrinsics)
        render.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        render.scene.set_view_size(size[0], size[1])
    
    render_scene = True # variable to indicate whether scene needs to be rendered again
    
    aria_data = data_loader(scan_dir, force_object_detection)
    context = deque(maxlen=8)

    # fill the context_window with observations
    while len(context) < context.maxlen:
        context.append(next(aria_data))

    left_positions, right_positions = deque(maxlen=10), deque(maxlen=10)
    

    object_poses, object_timestamps, camera_poses, camera_timestamps = [], [], [], []

    left_id, right_id = None, None
    left_prev_pose_inv, right_prev_pose_inv = None, None

    K = np.array([[611.43, 0, 703.5], [0, 611.43, 703.5], [0, 0, 1]])
    distCoeffs = np.zeros((5,1))

    for observation in aria_data:
        # process the first observation in the context window
        frame_id, camera_pose, hand_detections, object_detections, left_palm, right_palm, time_stamp, image = context[0]
        context.append(observation)

        if save_pose and camera_pose is not None:
            camera_poses.append(camera_pose)
            camera_timestamps.append(time_stamp)
        
        left_positions.append(left_palm)
        right_positions.append(right_palm)

        has_detection = check_hand_object_tracker(hand_detections, object_detections)
        
        left_vel_prior, left_vel_post = hand_velocity(left_positions), hand_velocity([obs[4] for obs in context])
        # nothing in left hand
        if left_id is None:
            if has_detection and left_palm is not None:
                dist, obj_id = scene_graph.nearest_node(left_palm)
                if dist < 0.1 and all([scene_graph.nearest_node(obs[4])[0] > dist  or (not check_hand_object_tracker(obs[2], obs[3])) for obs in context]) \
                    and sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(left_vel_prior-left_vel_post) > 0.025)): # important: left_palm is obs[4]
                    left_id = obj_id
                    print("left: object %d taken at frame %d" % (obj_id, frame_id))

                    # compute hand-object offset in object coordinates
                    delta_world_left = left_palm - scene_graph.nodes[left_id].pose[:3, 3]
                    delta_obj_left = scene_graph.nodes[left_id].pose[:3, :3].T @ delta_world_left

                    
                    # project object points to image for 2d-3d correspondences
                    left_queries, left_queries_3D = project_pcd_to_image(scene_graph.nodes[left_id].tracking_points, image, camera_pose, scan_dir)
                    
                    # transform object points to object coordinate system
                    left_queries_object = (scene_graph.nodes[left_id].pose[:3, :3].T @ (left_queries_3D - scene_graph.nodes[left_id].pose[:3, 3]).T).T
                

                    ### CoTracker initialization
                    scene_graph.left_cotracker = CoTrackerOnlinePredictor(checkpoint="thirdparty/cotracker/checkpoint/cotracker2.pth")
                    scene_graph.left_cotracker = scene_graph.left_cotracker.to(DEFAULT_DEVICE)
                    left_is_first_step = True
                    left_cotracker_index = 0
                    left_frames = []


                    left_queries = (torch.from_numpy(left_queries)[None]).to(DEFAULT_DEVICE)
                    left_frames.append(image)

                    left_cotracker_index += 1
                    for img in [ct[7] for ct in context]:
                        if left_cotracker_index % scene_graph.left_cotracker.step == 0 and left_cotracker_index != 0:
                            left_pred_tracks, left_pred_visibility = _process_step(
                                scene_graph.left_cotracker,
                                left_frames,
                                left_is_first_step,
                                queries=left_queries,
                            )
                            left_is_first_step = False
                        left_frames.append(img)
                        left_cotracker_index += 1
                    
                    obj_pose_left = np.eye(4)
                    left_ind = (left_cotracker_index - 1) % scene_graph.left_cotracker.step - 8
                    R_obj_cam, t_vec = compute_rotation(left_pred_tracks, left_pred_visibility, left_queries_object, left_ind, K, distCoeffs)
                    R_obj_world = camera_pose[:3, :3] @ R_obj_cam
                    
                    if use_hand:
                        t_obj_world = left_palm - R_obj_world @ delta_obj_left
                    else:
                        t_obj_world = camera_pose[:3, :3] @ t_vec + camera_pose[:3, 3]

                    obj_pose_left[:3, :3] = R_obj_world
                    obj_pose_left[:3, 3] = t_obj_world
                    
                    left_prev_pose_inv = np.linalg.inv(obj_pose_left)
                    
                    # object was already in right hand in previous iteration
                    if left_id == right_id:
                        scene_graph.transform(left_id, np.dot(obj_pose_left, right_prev_pose_inv))
                        render_scene=True
        else:
            if left_palm is not None :
                if sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(left_vel_prior-left_vel_post) > 0.025)):

                    ### cotracker:
                    if left_cotracker_index % scene_graph.left_cotracker.step == 0 and left_cotracker_index != 0:
                        left_pred_tracks, left_pred_visibility = _process_step(
                            scene_graph.left_cotracker,
                            left_frames,
                            left_is_first_step,
                            queries=left_queries,
                        )
                        left_is_first_step = False
                    left_frames.append(context[-1][7])
                    left_cotracker_index += 1
                    
                    obj_pose_left = np.eye(4)
                    left_ind = (left_cotracker_index - 1) % scene_graph.left_cotracker.step - 8
                    R_obj_cam, t_vec = compute_rotation(left_pred_tracks, left_pred_visibility, left_queries_object, left_ind, K, distCoeffs)
                    R_obj_world = camera_pose[:3, :3] @ R_obj_cam
                    
                    if use_hand:
                        t_obj_world = left_palm - R_obj_world @ delta_obj_left
                    else:
                        t_obj_world = camera_pose[:3, :3] @ t_vec + camera_pose[:3, 3]

                    obj_pose_left[:3, :3] = R_obj_world
                    obj_pose_left[:3, 3] = t_obj_world
                    
                    scene_graph.transform(left_id, np.dot(obj_pose_left, left_prev_pose_inv))
                    render_scene = True

                    left_prev_pose_inv = np.linalg.inv(obj_pose_left)
                else:
                    print("left: object %d released at frame %d" % (left_id, frame_id))
                    left_id = None
                    left_prev_pose_inv = None
                    obj_pose_left = None
                    if video_path:
                        len_tracks = left_pred_tracks.shape[1]
                        video = torch.tensor(np.stack(left_frames[:len_tracks]), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
                        vis = Visualizer(save_dir=os.path.dirname(video_path),linewidth=3, fps=30)
                        vis.visualize(video, left_pred_tracks, left_pred_visibility, filename="left_cotracker")
                    scene_graph.left_cotracker = None
        
        
        right_vel_prior, right_vel_post = hand_velocity(right_positions), hand_velocity([obs[5] for obs in context])
        if right_id is None:
            if has_detection and right_palm is not None:
                dist, obj_id = scene_graph.nearest_node(right_palm)
                if dist < 0.1 and all([scene_graph.nearest_node(obs[5])[0] > dist or (not check_hand_object_tracker(obs[2], obs[3])) for obs in context]) \
                    and sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(right_vel_prior-right_vel_post) > 0.025)): # important: right_palm is obs[5]
                    right_id = obj_id
                    print("right: object %d taken at frame %d" % (obj_id, frame_id))

                    # compute hand-object offset in object coordinates
                    delta_world_right = right_palm - scene_graph.nodes[right_id].pose[:3, 3]
                    delta_obj_right = scene_graph.nodes[right_id].pose[:3, :3].T @ delta_world_right


                    # project object points to image for 2d-3d correspondences
                    right_queries, right_queries_3D = project_pcd_to_image(scene_graph.nodes[right_id].tracking_points, image, camera_pose, scan_dir)
                    
                    # transform object points to object coordinate system
                    right_queries_object = (scene_graph.nodes[right_id].pose[:3, :3].T @ (right_queries_3D - scene_graph.nodes[right_id].pose[:3, 3]).T).T
                    
                    ### CoTracker initialization
                    scene_graph.right_cotracker = CoTrackerOnlinePredictor(checkpoint="thirdparty/cotracker/checkpoint/cotracker2.pth")
                    scene_graph.right_cotracker = scene_graph.right_cotracker.to(DEFAULT_DEVICE)
                    right_is_first_step = True
                    right_cotracker_index = 0
                    right_frames = []

                    right_queries = (torch.from_numpy(right_queries)[None]).to(DEFAULT_DEVICE)
                    right_frames.append(image)

                    right_cotracker_index += 1
                    for img in [ct[7] for ct in context]:
                        if right_cotracker_index % scene_graph.right_cotracker.step == 0 and right_cotracker_index != 0:
                            right_pred_tracks, right_pred_visibility = _process_step(
                                scene_graph.right_cotracker,
                                right_frames,
                                right_is_first_step,
                                queries=right_queries,
                            )
                            right_is_first_step = False
                        right_frames.append(img)
                        right_cotracker_index += 1

                    obj_pose_right = np.eye(4)
                    right_ind = (right_cotracker_index - 1) % scene_graph.right_cotracker.step - 8
                    R_obj_cam, t_vec = compute_rotation(right_pred_tracks, right_pred_visibility, right_queries_object, right_ind, K, distCoeffs)
                    R_obj_world = camera_pose[:3, :3] @ R_obj_cam
                    
                    if use_hand: 
                        t_obj_world = right_palm - R_obj_world @ delta_obj_right
                    else:
                        t_obj_world = camera_pose[:3, :3] @ t_vec + camera_pose[:3, 3]

                    obj_pose_right[:3, :3] = R_obj_world
                    obj_pose_right[:3, 3] = t_obj_world

                    right_prev_pose_inv = np.linalg.inv(obj_pose_right)
        else:
            if right_palm is not None:
                if sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(right_vel_prior-right_vel_post) > 0.025)):

                    if right_cotracker_index % scene_graph.right_cotracker.step == 0 and right_cotracker_index != 0:
                        right_pred_tracks, right_pred_visibility = _process_step(
                            scene_graph.right_cotracker,
                            right_frames,
                            right_is_first_step,
                            queries=right_queries,
                        )
                        right_is_first_step = False
                    right_frames.append(context[-1][7])
                    right_cotracker_index += 1
                    
                    obj_pose_right = np.eye(4)
                    right_ind = (right_cotracker_index - 1) % scene_graph.right_cotracker.step - 8
                    R_obj_cam, t_vec = compute_rotation(right_pred_tracks, right_pred_visibility, right_queries_object, right_ind, K, distCoeffs)
                    R_obj_world = camera_pose[:3, :3] @ R_obj_cam
                    
                    if use_hand:
                        t_obj_world = right_palm - R_obj_world @ delta_obj_right
                    else:
                        t_obj_world = camera_pose[:3, :3] @ t_vec + camera_pose[:3, 3]

                    obj_pose_right[:3, :3] = R_obj_world
                    obj_pose_right[:3, 3] = t_obj_world

                    if right_id != left_id:
                        scene_graph.transform(right_id, np.dot(obj_pose_right, right_prev_pose_inv))
                        render_scene = True

                    right_prev_pose_inv = np.linalg.inv(obj_pose_right)
                else:
                    print("right: object %d released at frame %d" % (right_id, frame_id))
                    right_id = None
                    right_prev_pose_inv = None
                    delta_obj_right = None
                    if video_path:
                        len_tracks = right_pred_tracks.shape[1]
                        video = torch.tensor(np.stack(right_frames[:len_tracks]), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
                        vis = Visualizer(save_dir=os.path.dirname(video_path),linewidth=3, fps=30)
                        vis.visualize(video, right_pred_tracks, right_pred_visibility, filename="right_cotracker")
                    scene_graph.right_cotracker = None
            
        if save_pose:
            if left_id is not None:
                if tracker_offset is not None:
                    tmp_pose = np.eye(4)
                    tmp_pose[:3,:3] = scene_graph.nodes[left_id].pose[:3,:3]
                    tmp_pose[:3,3] = scene_graph.nodes[left_id].pose[:3,:3] @ tracker_offset + scene_graph.nodes[left_id].pose[:3,3]
                    object_poses.append(tmp_pose)
                else:
                    object_poses.append(scene_graph.nodes[left_id].pose)
                object_timestamps.append(time_stamp)
            if right_id is not None and right_id != left_id:
                if tracker_offset is not None:
                    tmp_pose = np.eye(4)
                    tmp_pose[:3,:3] = scene_graph.nodes[right_id].pose[:3,:3]
                    tmp_pose[:3,3] = scene_graph.nodes[right_id].pose[:3,:3] @ tracker_offset + scene_graph.nodes[right_id].pose[:3,3]
                    object_poses.append(tmp_pose)
                else:
                    object_poses.append(scene_graph.nodes[right_id].pose)
                object_timestamps.append(time_stamp)
        
        if video_path:
            def create_hand(position, id, left_hand):
                hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                hand.translate(position)
                if left_hand:
                    if id: hand.paint_uniform_color([0.898, 0.224, 0.208])
                    else: hand.paint_uniform_color([0.298, 0.686, 0.314])
                else:
                    if id: hand.paint_uniform_color([0.470, 0.368, 0.941])
                    else: hand.paint_uniform_color([0.996, 0.380, 0])
                hand.compute_vertex_normals()
                return hand
            
            render.scene.remove_geometry("left_"+str(frame_id-10))
            render.scene.remove_geometry("right_"+str(frame_id-10))

            if render_scene:
                render.scene.clear_geometry()
                geometries = scene_graph.scene_geometries()

                for geometry, name, material in geometries:
                    render.scene.add_geometry(name, geometry, material)

                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
                
                for i, left_pos in enumerate(left_positions):
                    if left_pos is None: continue
                    left_hand = create_hand(left_pos, left_id, True)
                    render.scene.add_geometry("left_"+str(frame_id-9+i), left_hand, material)
                for i, right_pos in enumerate(right_positions):
                    if right_pos is None: continue
                    right_hand = create_hand(right_pos, right_id, False)
                    render.scene.add_geometry("right_"+str(frame_id-9+i), right_hand, material)
                
                if save_pose and object_poses:
                    centroid_pose = object_poses[-1]
                    centroid_pos = centroid_pose[:3,3]
                    # centroid_pos = scene_graph.nodes[right_id].centroid
                    centroid_ = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                    centroid_.translate(centroid_pos)
                    centroid_.paint_uniform_color([1, 0.0, 0.0])
                    centroid_.compute_vertex_normals()
                    render.scene.add_geometry("centroid"+str(frame_id), centroid_, material)

            else:
                if left_palm is not None:
                    left_hand = create_hand(left_palm, left_id, True)
                    render.scene.add_geometry("left_"+str(frame_id), left_hand, material)
                        
                if right_palm is not None:
                    right_hand = create_hand(right_palm, right_id, False)
                    render.scene.add_geometry("right_"+str(frame_id), right_hand, material)
            
            images += [render.render_to_image()]

            real_images += [image]
        
        render_scene = False


    
    if video_path:
        # create video from images
        with imageio.get_writer(video_path, fps=30, macro_block_size=None) as writer:
            for image in tqdm(images):
                writer.append_data(np.asarray(image))
    
    if save_pose:
        prefix = ""
        if not use_hand: prefix = "pose_only_"
        np.save(os.path.join(scan_dir, prefix + "glasses_trajectory.npy"), np.array(camera_poses))
        np.save(os.path.join(scan_dir, prefix + "glasses_timestamps.npy"), np.array(camera_timestamps))
        np.save(os.path.join(scan_dir, prefix + "object_trajectory.npy"), np.array(object_poses))
        np.save(os.path.join(scan_dir, prefix + "object_timestamps.npy"), np.array(object_timestamps))


def headpose_track(
    scene_graph: SceneGraph,
    scan_dir: str,
    video_path: Optional[str] = None,
    force_object_detection: bool = False,
    save_pose: bool = False,
    tracker_offset: Optional[np.ndarray] = None,
    use_hand: bool = True
) -> None:
    """ Employs same strategy as track method but with headpose heuristic for the object rotation instead of point tracking.
    Served as a baseline in the experiments. Less accurate but more robust to non-rigid objects. Consider this if your applications
    involve non-rigid objects.

    :param scene_graph: The scene graph containing object and spatial information.
    :param scan_dir: Directory containing scan data and where results will be saved.
    :param video_path: Path where tracking visualization video will be saved.
    :param force_object_detection: Whether to force object detection in each frame.
    :param save_pose: Whether to save tracked poses to disk.
    :param tracker_offset: Optional offset to apply to tracked poses.
    :param use_hand: Whether to use hand positions for pose estimation.
    :return: None   
    """    
    if video_path:
        images = []

        intrinsics, extrinsics = scene_graph.set_camera()
        render = o3d.visualization.rendering.OffscreenRenderer(1296, 1296)
        geometries = scene_graph.scene_geometries()
        for geometry, name, material in geometries:
            render.scene.add_geometry(name, geometry, material)
        render.setup_camera(intrinsics, extrinsics)
        render.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        render.scene.set_view_size(1296, 1296)
    
    aria_data = data_loader(scan_dir, force_object_detection)
    context = deque(maxlen=8)
    render_scene = True

    # fill the context_window with observations
    while len(context) < context.maxlen:
        context.append(next(aria_data))
    
    left_positions, right_positions = deque(maxlen=10), deque(maxlen=10)

    object_poses, object_timestamps, camera_poses, camera_timestamps = [], [], [], []


    left_id, right_id = None, None
    left_prev_pose_inv, right_prev_pose_inv = None, None
    
    window_frames = []

    for observation in aria_data:
        # process the first observation in the context window
        frame_id, camera_pose, hand_detections, object_detections, left_palm, right_palm, time_stamp, image = context[0]
        context.append(observation)

        if camera_pose is not None:
            camera_poses.append(camera_pose)
            camera_timestamps.append(time_stamp)
        
        left_positions.append(left_palm)
        right_positions.append(right_palm)

        has_detection = check_hand_object_tracker(hand_detections, object_detections)
        left_vel_prior, left_vel_post = hand_velocity(left_positions), hand_velocity([obs[4] for obs in context])
        # nothing in left hand
        if left_id is None:
            if has_detection and left_palm is not None:
                dist, obj_id = scene_graph.nearest_node(left_palm)
                if dist < 0.1 and all([scene_graph.nearest_node(obs[4])[0] > dist  or (not check_hand_object_tracker(obs[2], obs[3])) for obs in context]) \
                    and sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(left_vel_prior-left_vel_post) > 0.025)): # important: left_palm is obs[4]
                    left_id = obj_id
                    print("left: object %d taken at frame %d" % (obj_id, frame_id))

                    # compute hand-object offset in object coordinates
                    delta_world_left = left_palm - scene_graph.nodes[left_id].pose[:3, 3]
                    delta_obj_left = scene_graph.nodes[left_id].pose[:3, :3].T @ delta_world_left
                    
                    obj_pose_left = np.eye(4)

                    R_obj_world = camera_pose[:3, :3]
                    t_obj_world = left_palm - R_obj_world @ delta_obj_left

                    obj_pose_left[:3, :3] = R_obj_world
                    obj_pose_left[:3, 3] = t_obj_world
                    
                    left_prev_pose_inv = np.linalg.inv(obj_pose_left)

                    if left_id == right_id:
                        scene_graph.transform(left_id, np.dot(obj_pose_left, right_prev_pose_inv))
                        render_scene = True
        else:
            if left_palm is not None :
                if sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(left_vel_prior-left_vel_post) > 0.025)):

                    obj_pose_left = np.eye(4)

                    R_obj_world = camera_pose[:3, :3]
                    t_obj_world = left_palm - R_obj_world @ delta_obj_left

                    obj_pose_left[:3, :3] = R_obj_world
                    obj_pose_left[:3, 3] = t_obj_world

                    scene_graph.transform(left_id, np.dot(obj_pose_left, left_prev_pose_inv))                        
                    render_scene=True
                    
                    left_prev_pose_inv = np.linalg.inv(obj_pose_left)
                else:
                    print("left: object %d released at frame %d" % (left_id, frame_id))
                    left_id = None
                    left_prev_pose_inv = None
        
        
        right_vel_prior, right_vel_post = hand_velocity(right_positions), hand_velocity([obs[5] for obs in context])
        if right_id is None:
            if has_detection and right_palm is not None:
                dist, obj_id = scene_graph.nearest_node(right_palm)
                if dist < 0.1 and all([scene_graph.nearest_node(obs[5])[0] > dist or (not check_hand_object_tracker(obs[2], obs[3])) for obs in context]) \
                    and sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(right_vel_prior-right_vel_post) > 0.025)): # important: right_palm is obs[5]
                    right_id = obj_id
                    print("right: object %d taken at frame %d" % (obj_id, frame_id))

                    # compute hand-object offset in object coordinates
                    delta_world_right = right_palm - scene_graph.nodes[right_id].pose[:3, 3]
                    delta_obj_right = scene_graph.nodes[right_id].pose[:3, :3].T @ delta_world_right
                    
                    obj_pose_right = np.eye(4)

                    R_obj_world = camera_pose[:3, :3]
                    t_obj_world = right_palm - R_obj_world @ delta_obj_right

                    obj_pose_right[:3, :3] = R_obj_world
                    obj_pose_right[:3, 3] = t_obj_world
                    
                    right_prev_pose_inv = np.linalg.inv(obj_pose_right)
        else:
            if right_palm is not None:
                if sum([check_hand_object_tracker(context[i][2], context[i][3]) for i in range(len(context))]) > (3 + 2*(abs(right_vel_prior-right_vel_post) > 0.025)):

                    obj_pose_right = np.eye(4)

                    R_obj_world = camera_pose[:3, :3]
                    t_obj_world = right_palm - R_obj_world @ delta_obj_right

                    obj_pose_right[:3, :3] = R_obj_world
                    obj_pose_right[:3, 3] = t_obj_world

                    if right_id != left_id:
                        scene_graph.transform(right_id, np.dot(obj_pose_right, right_prev_pose_inv))                        
                        render_scene=True
                    
                    right_prev_pose_inv = np.linalg.inv(obj_pose_right)
                else:
                    print("right: object %d released at frame %d" % (right_id, frame_id))
                    right_id = None
                    right_prev_pose_inv = None
        
        if left_id is not None or right_id is not None:
            window_frames.append(image)
        
        if save_pose:
            if left_id is not None:
                if tracker_offset is not None:
                    tmp_pose = np.eye(4)
                    tmp_pose[:3,:3] = scene_graph.nodes[left_id].pose[:3,:3]
                    tmp_pose[:3,3] = scene_graph.nodes[left_id].pose[:3,:3] @ tracker_offset + scene_graph.nodes[left_id].pose[:3,3]
                    object_poses.append(tmp_pose)
                else:
                    object_poses.append(scene_graph.nodes[left_id].pose)
                object_timestamps.append(time_stamp)
            if right_id is not None and right_id != left_id:
                if tracker_offset is not None:
                    tmp_pose = np.eye(4)
                    tmp_pose[:3,:3] = scene_graph.nodes[right_id].pose[:3,:3]
                    tmp_pose[:3,3] = scene_graph.nodes[right_id].pose[:3,:3] @ tracker_offset + scene_graph.nodes[right_id].pose[:3,3]
                    object_poses.append(tmp_pose)
                else:
                    object_poses.append(scene_graph.nodes[right_id].pose)
                object_timestamps.append(time_stamp)
        
        if video_path:
            def create_hand(position, id, left_hand):
                hand = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                hand.translate(position)
                if left_hand:
                    if id: hand.paint_uniform_color([0.898, 0.224, 0.208])
                    else: hand.paint_uniform_color([0.298, 0.686, 0.314])
                else:
                    if id: hand.paint_uniform_color([0.470, 0.368, 0.941])
                    else: hand.paint_uniform_color([0.996, 0.380, 0])
                hand.compute_vertex_normals()
                return hand
            
            render.scene.remove_geometry("left_"+str(frame_id-10))
            render.scene.remove_geometry("right_"+str(frame_id-10))

            if render_scene:
                render.scene.clear_geometry()
                geometries = scene_graph.scene_geometries()

                for geometry, name, material in geometries:
                    render.scene.add_geometry(name, geometry, material)

                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
                
                for i, left_pos in enumerate(left_positions):
                    if left_pos is None: continue
                    left_hand = create_hand(left_pos, left_id, True)
                    render.scene.add_geometry("left_"+str(frame_id-9+i), left_hand, material)
                for i, right_pos in enumerate(right_positions):
                    if right_pos is None: continue
                    right_hand = create_hand(right_pos, right_id, False)
                    render.scene.add_geometry("right_"+str(frame_id-9+i), right_hand, material)
                
                if object_poses:
                    centroid_pose = object_poses[-1]
                    centroid_pos = centroid_pose[:3,3]
                    centroid_ = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                    centroid_.translate(centroid_pos)
                    centroid_.paint_uniform_color([1, 0.0, 0.0])
                    centroid_.compute_vertex_normals()
                    render.scene.add_geometry("centroid"+str(frame_id), centroid_, material)

            else:
                if left_palm is not None:
                    left_hand = create_hand(left_palm, left_id, True)
                    render.scene.add_geometry("left_"+str(frame_id), left_hand, material)
                        
                if right_palm is not None:
                    right_hand = create_hand(right_palm, right_id, False)
                    render.scene.add_geometry("right_"+str(frame_id), right_hand, material)
            
            images += [render.render_to_image()]
        
        render_scene = False
        
    if save_pose:
        filename = "headpose"
        if not use_hand: filename = "pose_only_" + filename
        np.save(scan_dir + "/" + filename + "_glasses_trajectory.npy", np.array(camera_poses))
        np.save(scan_dir + "/" + filename + "_glasses_timestamps.npy", np.array(camera_timestamps))
        np.save(scan_dir + "/" + filename + "_object_trajectory.npy", np.array(object_poses))
        np.save(scan_dir + "/" + filename + "_object_timestamps.npy", np.array(object_timestamps))
    
    if video_path:
        # create video from images
        with imageio.get_writer(video_path, fps=30) as writer:
            for image in tqdm(images):
                writer.append_data(np.asarray(image))