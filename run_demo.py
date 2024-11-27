import numpy as np
import pandas as pd
from src import SceneGraph, preprocess_scan, preprocess_aria


if __name__ == "__main__":
    SCAN_DIR = "Data/Scan"
    ARIA_DATA = "Data/Scene_name"

    # instantiate the label mapping for Mask3D object classes
    label_map = pd.read_csv('mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    
    # does the preprocessing (if not applied before)
    preprocess_scan(SCAN_DIR, drawer_detection=False, light_switch_detection=False)
    preprocess_aria(SCAN_DIR, ARIA_DATA)
    
    T_aria = np.load(ARIA_DATA + "/aruco_pose.npy")
    T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
    
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, immovable=["armchair", "bookshelf", "end table", "shelf", "coffee table", "dresser"], pose=T_aria)
    
    ### builds the scene_graph from Mask3D output
    scene_graph.build(SCAN_DIR, ARIA_DATA, drawers=False, light_switches=False)
    
    ### for nicer visuals, remove certain categories within your scene, of course optional:
    scene_graph.remove_category("curtain")
    scene_graph.remove_category("door")

    ### visualizes the current state of the scene graph with different visualizaion options:
    scene_graph.visualize(centroids=True, connections=True, labels=True)
    
    ### applies the tracked interactions from the ARIA_DATA to the scene graph datastructure:
    scene_graph.track(ARIA_DATA, ARIA_DATA + "/tracking.mp4")

    ### the changes are saved within the datastructure, so visualizing the scene graph always shows its current state:
    scene_graph.visualize()
        