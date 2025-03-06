import torch
import numpy as np
import pandas as pd
import os
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
from trajectory_utils.trajectory import Trajectory
from trajectory_utils.io.np_reader import trajectory_from_numpy
from trajectory_utils.io.bag_converter import load_trajectories_from_bag
from trajectory_utils.utils.colors import get_colormap
import plotly.graph_objects as go


def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    R_gt, R_pred = R.from_quat(R_gt).as_matrix(), R.from_quat(R_pred).as_matrix()

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] @ pts3d.transpose() + t_gt[i].reshape(3, 1)
        pts_xformed_pred = R_pred[i] @ pts3d.transpose() + t_pred[i].reshape(3, 1)
        kdt = KDTree(pts_xformed_gt.transpose())
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    R_gt, R_pred = R.from_quat(R_gt).as_matrix(), R.from_quat(R_pred).as_matrix()


    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] @ pts3d.transpose() + t_gt[i].reshape(3, 1)
        pts_xformed_pred = R_pred[i] @ pts3d.transpose() + t_pred[i].reshape(3, 1)
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)            

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_diameter(points_3D):
    hull = ConvexHull(points_3D)
    hull_points = points_3D[hull.vertices]

    distances = pdist(hull_points)
    return np.max(distances)


# Smaller helper function to load Prediction trajectories
def get_prediction_trajectories(
        path_to_object_trajectory: str, path_to_object_timestamps: str,
        path_to_glasses_trajectory: str, path_to_glasses_timestamps: str) -> dict[str, Trajectory]:

    """Load prediction trajectories from a numpy file.
    
    Args:
        path_to_prediction (str): Path to the numpy file.
        
    Returns:
        dict: A dictionary of trajectories.
    """
    pred_hand_traj = trajectory_from_numpy(path_to_object_trajectory, path_to_object_timestamps, child_frame="Prediction/RigidBody", parent_frame="world")
    pred_glasses_traj = trajectory_from_numpy(path_to_glasses_trajectory, path_to_glasses_timestamps, child_frame="RigidBody/AriaGlasses", parent_frame="world")
    return {"Prediction/RigidBody": pred_hand_traj, "RigidBody/AriaGlasses": pred_glasses_traj}

def trim_trajectories_to_interaction(pred_obj: Trajectory, pred_glasses: Trajectory, gt_obj: Trajectory, gt_glasses: Trajectory) -> tuple[Trajectory, Trajectory, Trajectory, Trajectory]:
    """Trim object trajectories based on object interaction.
    
    This function assumes the given trajectories to be time-synchronized.
    It then determines start and end of an interaction based on the object trajectory. If the object is moving more than 0.05 m/s, it is considered an interaction.
    Once the object does not move more than 0.05 m/s, the interaction is considered to be over.

    Args:
        pred_obj (Trajectory): Prediction object trajectory.
        pred_glasses (Trajectory): Prediction glasses trajectory.
        gt_obj (Trajectory): Ground truth object trajectory.
        gt_glasses (Trajectory): Ground truth glasses trajectory.
    
    Returns:
        tuple[Trajectory, Trajectory, Trajectory, Trajectory]: Trimmed trajectories, based on the interaction of the gt_obj trajectory.
    """
    sampling_frequency = 30
    time_sync_slack = 0.1

    # We use the object trajectory to find the start and end of the interaction
    positions = gt_obj.positions
    positions_1d = positions.norm(dim=1)
    positions_1d_diff = torch.cat([torch.zeros(1, device=positions_1d.device, dtype=positions_1d.dtype), positions_1d.diff() / gt_obj._timesteps.diff().abs() ])
    # slightly smoothen, lowpass filter
    positions_1d_diff = torch.from_numpy(np.convolve(positions_1d_diff.numpy(), np.ones(5)/5, mode="same"))
    # determine interaction as when the object is moving more than 0.05 m/s
    interaction = abs(positions_1d_diff) > 0.05
    start_idx = torch.where(interaction)[0][0].item()
    end_idx = torch.where(interaction)[0][-1].item()
    # find start and end of interaction
    start_ts = gt_obj._timesteps[start_idx].item() - time_sync_slack
    end_ts = gt_obj._timesteps[end_idx].item() + time_sync_slack
    # make sure end_ts is valid match with frequency
    end_ts = end_ts - (end_ts - start_ts) % (1/sampling_frequency)

    # Finally, slice all trajectories to the interaction
    pred_obj = pred_obj.slice(start_time=start_ts, end_time=end_ts).resample(frequency=sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    pred_glasses = pred_glasses.slice(start_time=start_ts, end_time=end_ts).resample(frequency=sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    gt_obj = gt_obj.slice(start_time=start_ts, end_time=end_ts).resample(frequency=sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    gt_glasses = gt_glasses.slice(start_time=start_ts, end_time=end_ts).resample(frequency=sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    return pred_obj, pred_glasses, gt_obj, gt_glasses


def trajectory_from_csv(
    csv_file: str,
    timestamp_unit: float = 1.0,
) -> Trajectory:
    """
    Reads a 6DoF trajectory from a CSV file and returns a Trajectory object.

    The CSV file is expected to have the following structure:
        # parent_frame: <parent_frame>
        # child_frame: <child_frame>
        timestamp, tx, ty, tz, qx, qy, qz, qw

    Args:
        csv_file (str): Path to the CSV file.
        timestamp_unit (float, optional): Factor to convert timestamps to seconds.
                                          Defaults to 1.0.

    Returns:
        Trajectory: A Trajectory object constructed from the CSV data.
    """
    parent_frame, child_frame = None, None

    # Read metadata from header
    with open(csv_file, "r") as f:
        lines = f.readlines()

    # Extract frame names from commented lines
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("# parent_frame:"):
            parent_frame = line.split(":")[1].strip()
        elif line.startswith("# child_frame:"):
            child_frame = line.split(":")[1].strip()
        else:
            data_start = i
            break  # Stop when actual data begins

    # Ensure we have frame names
    if parent_frame is None or child_frame is None:
        raise ValueError("CSV file is missing parent_frame or child_frame metadata.")

    # Read CSV data (skipping metadata lines)
    df = pd.read_csv(csv_file, skiprows=data_start)

    required_columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV file.")

    positions = torch.from_numpy(df[["tx", "ty", "tz"]].values).float()
    orientations = torch.from_numpy(df[["qx", "qy", "qz", "qw"]].values).float()
    timestamps = torch.from_numpy(df["timestamp"].values).double() / timestamp_unit

    return Trajectory(positions, orientations, timestamps, parent_frame, child_frame)


def trajectory_to_csv(
    trajectory: Trajectory,
    csv_file: str,
    timestamp_unit: float = 1.0,
) -> None:
    """
    Saves a Trajectory object as a CSV file, including metadata as header comments.

    The CSV file will have the following structure:
        # parent_frame: <parent_frame>
        # child_frame: <child_frame>
        timestamp, tx, ty, tz, qx, qy, qz, qw

    Args:
        trajectory (Trajectory): The Trajectory object to save.
        csv_file (str): The file path where the CSV will be saved.
        timestamp_unit (float, optional): Factor to convert timestamps from seconds to desired units.
                                          Defaults to 1.0.
    """
    timestamps = trajectory.timesteps * timestamp_unit
    positions = trajectory.positions
    orientations = trajectory.orientations

    df = pd.DataFrame({
        "timestamp": timestamps.cpu().numpy(),
        "tx": positions[:, 0].cpu().numpy(),
        "ty": positions[:, 1].cpu().numpy(),
        "tz": positions[:, 2].cpu().numpy(),
        "qx": orientations[:, 0].cpu().numpy(),
        "qy": orientations[:, 1].cpu().numpy(),
        "qz": orientations[:, 2].cpu().numpy(),
        "qw": orientations[:, 3].cpu().numpy(),
    })

    # Write metadata as commented lines at the top
    with open(csv_file, "w") as f:
        f.write(f"# parent_frame: {trajectory.parent_frame}\n")
        f.write(f"# child_frame: {trajectory.child_frame}\n")
        df.to_csv(f, index=False)

    print(f"Trajectory saved to {csv_file} with frame metadata.")


def create_GT(DATA_FOLDER, sampling_frequency, headless=True, show_object=False):
    """Generate ground truth object and glasses trajectories, aligning them with predictions.

    This function loads and processes ground truth and predicted trajectories from recorded
    data, aligns them spatially and temporally, transforms them into the appropriate reference
    frames, and trims them based on interaction duration. Optionally, it visualizes the aligned
    trajectories and object points before saving them as .csv and .npy files.

    Args:
        DATA_FOLDER (str): Path to the dataset folder containing scene results.
        sampling_frequency (int): Frequency for resampling the trajectories.
        headless (bool, optional): If False, visualizes aligned trajectories. Defaults to True.
        show_object (bool, optional): If True, displays additional 3D object points. Defaults to False.

    Returns:
        None: Saves processed ground truth and predicted trajectories in the dataset folder.
    """
    for scene in sorted(os.listdir(DATA_FOLDER)):
        scene_folder = os.path.join(DATA_FOLDER, scene)
        if not os.path.exists(os.path.join(scene_folder, "object_trajectory.npy")) or np.load(os.path.join(scene_folder, "object_trajectory.npy")).shape[0] == 0:
            print("Skipping scene, no predictions found")
            continue
        # Load data
        gt_trajectories = load_trajectories_from_bag(os.path.join(scene_folder, scene + ".bag"))
        pred_trajectories = get_prediction_trajectories(
            os.path.join(scene_folder, "object_trajectory.npy"), os.path.join(scene_folder, "object_timestamps.npy"),
            os.path.join(scene_folder, "glasses_trajectory.npy"), os.path.join(scene_folder, "glasses_timestamps.npy"),
        )

        # Resample the prediction trajectories at the same frequency
        for key, traj in pred_trajectories.items():
            pred_trajectories[key] = traj.resample(frequency=sampling_frequency)
        for key, traj in gt_trajectories.items():
            gt_trajectories[key] = traj.resample(frequency=sampling_frequency)

        gt_glasses_w, gt_object_w = gt_trajectories["vicon/AriaGlasses"], gt_trajectories["vicon/RigidBody"]
        pred_glasses_w, pred_object_w = pred_trajectories["RigidBody/AriaGlasses"], pred_trajectories["Prediction/RigidBody"]

        # align the head trajectories to find time delay and rotation
        gt_glasses_w, pred_glasses_w_aligned, infos = gt_glasses_w.clone().temporal_align(pred_glasses_w, return_infos=True)
        delay, rotation, translation = infos["delay"], infos["rotation"], infos["translation"]
        gt_object_w = gt_object_w.slice(start_time=gt_glasses_w.start_time, end_time=gt_glasses_w.end_time)

        # Compute inverse transformation to keep all data in Aria reference frame
        translation_np = translation.cpu().numpy()
        rotation_np = rotation.cpu().numpy()

        inv_rotation_scipy = R.from_quat(rotation_np).inv()
        inv_translation_np = -inv_rotation_scipy.apply(translation_np)

        inv_translation = torch.tensor(inv_translation_np, dtype=translation.dtype, device=translation.device)
        inv_rotation = torch.tensor(inv_rotation_scipy.as_quat(), dtype=rotation.dtype, device=rotation.device)

        # Shift all predictions to achieve spatial and temporal alignment
        pred_glasses_w = pred_glasses_w.transform(translation, rotation)
        pred_object_w = pred_object_w.transform(translation, rotation)
        pred_glasses_w.parent_frame = "vicon"
        pred_object_w.parent_frame = "vicon"
        pred_glasses_w._timesteps -= delay
        pred_object_w._timesteps -= delay

        # Trim the trajectories to match the object interaction
        pred_object_w, pred_glasses_w, gt_object_w, gt_glasses_w = trim_trajectories_to_interaction(pred_object_w, pred_glasses_w, gt_object_w, gt_glasses_w)

        # Bring back to Aria coordinate system
        pred_glasses_w = pred_glasses_w.transform(inv_translation, inv_rotation)
        pred_object_w = pred_object_w.transform(inv_translation, inv_rotation)
        gt_glasses_w = gt_glasses_w.transform(inv_translation, inv_rotation)
        gt_object_w = gt_object_w.transform(inv_translation, inv_rotation)

        # Match the orientation of GT and predicted trajectory, such that the first orientation is the same
        initial_pose_pred_obj = pred_object_w[0:1].clone()
        initial_pose_pred_obj._positions *= 0
        initial_pose_pred_obj.parent_frame = gt_object_w.child_frame

        initial_pose_gt_obj = gt_object_w[0:1].clone()
        initial_pose_gt_obj._positions *= 0
        initial_pose_gt_obj.parent_frame = gt_object_w.child_frame
        
        gt_object_w = gt_object_w @ initial_pose_gt_obj.inverse()
        gt_object_w = gt_object_w @ initial_pose_pred_obj
        gt_object_w.child_frame = 'vicon/RigidBody'
        gt_object_w.parent_frame = 'vicon'

        initial_pose_pred_glasses = pred_glasses_w[0:1].clone()
        initial_pose_pred_glasses._positions *= 0
        initial_pose_pred_glasses.parent_frame = gt_glasses_w.child_frame

        initial_pose_gt_glasses = gt_glasses_w[0:1].clone()
        initial_pose_gt_glasses._positions *= 0
        initial_pose_gt_glasses.parent_frame = gt_glasses_w.child_frame

        gt_glasses_w = gt_glasses_w @ initial_pose_gt_glasses.inverse()
        gt_glasses_w = gt_glasses_w @ initial_pose_pred_glasses
        gt_glasses_w.child_frame = 'vicon/AriaGlasses'

        # load points and transform to object coordinate system within the scene
        object_category = scene.split("_")[0]        
        icp_pose = np.load(os.path.join(scene_folder, "icp_aligned_pose.npy"))
        
        obj_points = np.load("Data/Final_Models/" + object_category + ".npy")

        obj_points = (np.hstack((obj_points, np.ones((obj_points.shape[0], 1)))) @ icp_pose.T)[:, :3]

        rot_quaternion = R.from_quat(gt_object_w.orientations[0].cpu().numpy())
        orientation_w = rot_quaternion.as_matrix()
        translation_w = gt_object_w.positions[0].cpu().numpy().reshape(-1, 1)

        init_pose = T = np.vstack((np.hstack((orientation_w, translation_w)), [0, 0, 0, 1]))
        init_pose_inv = np.linalg.inv(init_pose)
        obj_points = (np.hstack((obj_points, np.ones((obj_points.shape[0], 1)))) @ init_pose_inv.T)[:, :3]


        if not headless:
            # Show aligned trajectories
            fig = pred_object_w.show(show=False, line_color="blue", time_as_color=get_colormap("B71C1C"), show_frames=True)
            fig = gt_object_w.show(fig, show=False, line_color="darkblue", time_as_color=get_colormap("FF1744"), show_frames=True)
            fig = pred_glasses_w.show(fig, show=False, line_color="#00C853", time_as_color=get_colormap("00CCFF"), show_frames=True)
            fig = gt_glasses_w.show(fig, show=False, line_color="#00CCFF", time_as_color=get_colormap("00BBCC"), show_frames=True)

            if show_object:
                fig.add_trace(
                    go.Scatter3d(
                        x=obj_points[:, 0],
                        y=obj_points[:, 1],
                        z=obj_points[:, 2],
                        mode="markers",
                        marker=dict(size=5, color="red", opacity=0.8),
                        name="Additional 3D Points",
                    )
                )

            # Title
            fig.update_layout(title="Aligned Trajectories")
            fig.show()
        
        np.save(os.path.join(scene_folder, "object_points.npy"), obj_points)

        trajectory_to_csv(gt_object_w, os.path.join(scene_folder, "gt_object.csv"))
        trajectory_to_csv(gt_glasses_w, os.path.join(scene_folder, "gt_glasses.csv"))
        trajectory_to_csv(pred_object_w, os.path.join(scene_folder, "pred_object.csv"))
        trajectory_to_csv(pred_glasses_w, os.path.join(scene_folder, "pred_glasses.csv"))


if __name__ == "__main__":
    pass