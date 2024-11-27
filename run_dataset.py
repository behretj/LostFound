import numpy as np
import pandas as pd
import os
from src import SceneGraph, preprocess_scan, preprocess_aria
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset processing and tracking")
    parser.add_argument('--scan_dir', type=str, required=True, help='Path to the scan directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--headpose', action='store_true', help='Flag to switch to headpose mode')
    parser.add_argument('--save_pose', action='store_true', help='Flag to save the pose data from the evaluation.')
    parser.add_argument('--pose_only', action='store_true', help='Flag to only use the predicted pose for trajectory prediction')
    args = parser.parse_args()


    SCAN_DIR = args.scan_dir
    DATA_FOLDER = args.data_dir

    use_hand = not args.pose_only

    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    preprocess_scan(SCAN_DIR, drawer_detection=False, light_switch_detection=False)
    
    tracklets = {"ball": np.array([1.46, -0.32, -0.52]),
                    "frame": np.array([1.2, 0.53, -1.1]),
                    "basket": np.array([0.69, -0.24, -0.86]),
                    "carton": np.array([1.95, 0.53, 0.47]),
                    "plant": np.array([-1.34, 0.866, 0.483]),
                    "water": np.array([-0.86, -0.31, 2.16]),
                    "clock": np.array([-1.34, 0.90, 1.015]),
                    "organizer": np.array([1.72, 1.18, -1.06]),
                    "shoe": np.array([-0.15, -0.48, -0.72]),}

    for name in sorted(os.listdir(DATA_FOLDER)):
        try:
            res = preprocess_aria(SCAN_DIR, DATA_FOLDER + "/" + name)
        except:
            continue
        if res is None: 
            print("Skipped:", name)
            continue
        
        T_aria = np.load(DATA_FOLDER + "/" + name + "/aruco_pose.npy")
        T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
        
        scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.10, immovable=["armchair", "bookshelf", "cabinet", "coffee table"], pose=T_aria)
        
        scene_graph.build(SCAN_DIR, DATA_FOLDER + "/" + name)

        ### for the visuals:
        scene_graph.remove_category("curtain")

        object_category = name.split("_")[0]
        print("Object Category: ", object_category)

        T_icp_aligned = np.load(DATA_FOLDER + "/" + name + "/icp_aligned_pose.npy")
        tracklet_point = tracklets.get(object_category, None)
        tracklet = np.dot(T_icp_aligned, np.array([tracklet_point[0], tracklet_point[1], tracklet_point[2], 1]))[:3]
        
        obj_id = scene_graph.ids[scene_graph.tree.query(tracklet)[1]]

        tracker_offset = tracklet - scene_graph.nodes[obj_id].centroid

        tracker_offset = np.dot(scene_graph.nodes[obj_id].pose[:3, :3].T, tracker_offset)
        
        if args.headpose:
            scene_graph.headpose_track(DATA_FOLDER + "/" + name, save_pose=args.save_pose, use_hand=use_hand, tracker_offset=tracker_offset)
        else:
            scene_graph.track(DATA_FOLDER + "/" + name, save_pose=args.save_pose, tracker_offset=tracker_offset, use_hand=use_hand)
        
        