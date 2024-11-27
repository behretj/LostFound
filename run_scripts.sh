#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/cvg-robotics/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/cvg-robotics/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/cvg-robotics/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/cvg-robotics/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# conda activate lost_found

# # Own Method
# # python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --save_pose > /home/cvg-robotics/tjark_ws/growing_scene_graphs/regular_run.log 2>&1
# # python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --save_pose --pose_only > /home/cvg-robotics/tjark_ws/growing_scene_graphs/pose_run.log 2>&1

# # Data generation + headpose baseline
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_data > /home/cvg-robotics/tjark_ws/growing_scene_graphs/headpose_bundle_data_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_data --foundation_data > /home/cvg-robotics/tjark_ws/growing_scene_graphs/headpose_foundation_data_run.log 2>&1

# conda deactivate

# # Depth creation
# conda activate metric3d

# python /home/cvg-robotics/tjark_ws/Metric3D/hubconf.py --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset_data > /home/cvg-robotics/tjark_ws/growing_scene_graphs/depth_data.log 2>&1
# # python /home/cvg-robotics/tjark_ws/Metric3D/hubconf.py --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset_foundationpose > /home/cvg-robotics/tjark_ws/growing_scene_graphs/depth_foundationpose.log 2>&1

# conda deactivate

# # Mask creation
# conda activate sam

# python /home/cvg-robotics/tjark_ws/sam2/mask_creation.py --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset_data > /home/cvg-robotics/tjark_ws/growing_scene_graphs/sam.log 2>&1

# conda deactivate


# # Wait for BundleSDF to finish
# echo "done" > /home/cvg-robotics/tjark_ws/growing_scene_graphs/done.flag

# FLAG_FILE="/home/cvg-robotics/tjark_ws/BundleSDF/done.flag"

# # Wait until the flag file is created
# while [ ! -f "$FLAG_FILE" ]; do
#     sleep 500  # Wait for 1 second before checking again
# done

conda activate lost_found

# ## Evaluating the baselines
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode BundleSDF > /home/cvg-robotics/tjark_ws/growing_scene_graphs/bundlesdf_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode BundleSDF --pose_only > /home/cvg-robotics/tjark_ws/growing_scene_graphs/pose_bundlesdf_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode FoundationPose > /home/cvg-robotics/tjark_ws/growing_scene_graphs/foundationpose_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode FoundationPose --pose_only > /home/cvg-robotics/tjark_ws/growing_scene_graphs/pose_foundationpose_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode BundleTrack > /home/cvg-robotics/tjark_ws/growing_scene_graphs/bundletrack_run.log 2>&1
# python /home/cvg-robotics/tjark_ws/growing_scene_graphs/run_dataset.py --scan_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Scan --data_dir /home/cvg-robotics/tjark_ws/growing_scene_graphs/Data/Final_Dataset --eval --save_pose --eval_mode BundleTrack --pose_only > /home/cvg-robotics/tjark_ws/growing_scene_graphs/pose_bundletrack_run.log 2>&1

conda deactivate
