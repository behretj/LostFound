from __future__ import annotations

from trajectory_utils.eval import compare_trajectories
from trajectory_utils.utils.colors import get_colormap
from eval_utils import compute_add_score, compute_adds_score, compute_diameter
from eval_utils import trajectory_from_csv
import os

import numpy as np
import json

import numpy as np


####################################################
# Main Script
# This can be used to evaluate the final predictions
####################################################

if __name__ == "__main__":
    DATA_FOLDER = "Data/Final_Dataset"
    headless = True
    add_score = True
    
    for scene in sorted(os.listdir(DATA_FOLDER)):
        scene_folder = os.path.join(DATA_FOLDER, scene)

        object_category = scene.split("_")[0]
        print("Object Category: ", object_category)
        
        obj_points = None
        if add_score:
            obj_points = np.load(os.path.join(scene_folder, "object_points.npy"))

        gt_object_w = trajectory_from_csv(os.path.join(scene_folder, "gt_object.csv"))
        pred_object_w = trajectory_from_csv(os.path.join(scene_folder, "pred_object.csv"))
        

        if not headless:
            # Show aligned trajectories
            fig = pred_object_w.show(show=False, line_color="blue", time_as_color=get_colormap("B71C1C"), show_frames=True)
            fig = gt_object_w.show(fig, show=False, line_color="darkblue", time_as_color=get_colormap("FF1744"), show_frames=True)
            # Title
            fig.update_layout(title="Aligned Trajectories")
            fig.show()

        data = {}
        # Eval needed quantities
        print("===== Object in World Frame =====")
        # Overwrite frame to make sure the comparison does not complain about different frames
        pred_object_w.child_frame = "vicon/RigidBody"
        data["object_metrics"] = compare_trajectories(gt_object_w, pred_object_w, headless=headless)
        
        if obj_points is not None:
            print("===== ADD Score =====")
            data["object_metrics"]["add_score"] = compute_add_score(obj_points, compute_diameter(obj_points), (gt_object_w.orientations.cpu().numpy(), gt_object_w.positions.cpu().numpy()), (pred_object_w.orientations.cpu().numpy(), pred_object_w.positions.cpu().numpy()))
            print("ADD Score:", data["object_metrics"]["add_score"])

            print("===== ADD-S Score =====")
            data["object_metrics"]["adds_score"] = compute_adds_score(obj_points, compute_diameter(obj_points), (gt_object_w.orientations.cpu().numpy(), gt_object_w.positions.cpu().numpy()), (pred_object_w.orientations.cpu().numpy(), pred_object_w.positions.cpu().numpy()))
            print("ADD-S Score:", data["object_metrics"]["adds_score"])


        # save the data
        with open(os.path.join(scene_folder, "results.json"), "w") as f:
            json.dump(data, f, indent=4)
            print(f"Saved evaluation for scene:", scene)