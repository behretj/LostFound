import os, json
import numpy as np
import os
import json
import numpy as np

def main_table(DATA_FOLDER="Data/Final_Dataset", prefixes=["headpose_", "BundleTrack_", "FoundationPose_", "BundleSDF_", ""], highlight_best=False):
    """Generate a LaTeX table summarizing RMSE position, RMSE rotation, and ADD-S scores for different methods.

    This function iterates through available scene data, extracting RMSE position errors,
    RMSE rotation errors, and ADD-S scores for multiple object categories. It then averages
    these values per object and across all objects to generate a formatted LaTeX table.

    Args:
        DATA_FOLDER (str, optional): Path to the dataset folder containing scene results. Defaults to "Data/Final_Dataset".
        prefixes (list, optional): List of prefixes representing different methods. Defaults to common methods.
        highlight_best (bool, optional): If True, highlights the best-performing method. Defaults to False.

    Returns:
        None: Prints the generated LaTeX table.
    """
    key_to_object = {
        'carton': 'Carton',
        'organizer': 'Organizer',
        'frame': 'Frame',
        'clock': 'Clock',
        'ball': 'Ball',
        'plant': 'Plant',
        'basket': 'Basket',
        'water': 'Water Can',
        'shoe': 'Shoe'
    }
    
    # Initialize dictionary to store LaTeX rows for each object
    table_rows = {key: [] for key in key_to_object}
    mean_values = []  # List to store mean values for each prefix

    for prefix in prefixes:
        rmse_positions = {key: [] for key in key_to_object}
        rmse_rotations = {key: [] for key in key_to_object}
        adds_scores = {key: [] for key in key_to_object}
        
        # Check if the JSON file for this prefix is available for each object
        for scene in sorted(os.listdir(DATA_FOLDER)):
            scene_folder = os.path.join(DATA_FOLDER, scene)
            object_category = scene.split("_")[0]

            # Load data only if the specific experiment file exists
            if os.path.exists(os.path.join(scene_folder, prefix + "results.json")):
                json_file = os.path.join(scene_folder, prefix + "results.json")
                json_data = json.load(open(json_file))

                rmse_position = json_data['object_metrics']['rmse_position'] * 100  # Convert to cm
                rmse_rotation = json_data['object_metrics']['rmse_rotation']
                rmse_rotation_deg = rmse_rotation * (180 / np.pi)  # Convert to degrees
                if 'add_score' in json_data['object_metrics']:
                    adds_score = json_data['object_metrics']['add_score'] * 100  # Convert to percentage

                if object_category in rmse_positions:
                    rmse_positions[object_category].append(rmse_position)
                    rmse_rotations[object_category].append(rmse_rotation_deg)
                    adds_scores[object_category].append(adds_score)
        
        # Calculate average values for each object and prepare LaTeX row values
        position_means, rotation_means, adds_means = [], [], []
        for key in key_to_object:
            if rmse_positions[key]:
                avg_position = sum(rmse_positions[key]) / len(rmse_positions[key])
                avg_rotation = sum(rmse_rotations[key]) / len(rmse_rotations[key])
                avg_adds = sum(adds_scores[key]) / len(adds_scores[key])
                position_means.append(avg_position)
                rotation_means.append(avg_rotation)
                adds_means.append(avg_adds)
            else:
                avg_position, avg_rotation, avg_adds = None, None, None
            
            table_rows[key].append((avg_position, avg_rotation, avg_adds))
        
        # Calculate and store mean values for the current prefix
        if position_means and rotation_means and adds_means:
            mean_position = f"${np.mean(position_means):.2f}$"
            mean_rotation = f"${np.mean(rotation_means):.2f}$"
            mean_adds = f"${np.mean(adds_means):.2f}\%$"
        else:
            mean_position, mean_rotation, mean_adds = "$--$", "$--$", "$--$"
        
        mean_values.append(f"{mean_position} & {mean_rotation} & {mean_adds}")
        
    latex_content = ""
    for key in key_to_object:
        object_name = key_to_object[key]
        metrics = table_rows[key]
        
        row_content = ""
        for pos, rot, adds in metrics:
            pos_str = f"${pos:.2f}$" if pos is not None else "$--$"
            rot_str = f"${rot:.2f}$" if rot is not None else "$--$"
            adds_str = f"${adds:.2f}\%$" if adds is not None else "$--$"
            row_content += f"{pos_str} & {rot_str} & {adds_str} & "
        
        latex_content += f"{object_name} & {row_content.strip('& ')} \\\\ \n"
    
    # Add the mean row at the end with \hline separation
    latex_content += "\\hline\n\\textbf{Mean} & " + " & ".join(mean_values) + " \\\\ \n"
    
    print(latex_content)

def summary_table(DATA_FOLDER="Data/Final_Dataset", prefixes=["headpose_", "BundleTrack_", "FoundationPose_", "BundleSDF_", ""]):
    """Generate a summary LaTeX table for median and standard deviation of translation and rotation errors.

    This function processes all experiment results for specified prefixes, computing median
    and standard deviation for RMSE translation and RMSE rotation errors. It also computes
    the ADD-S score and accuracy within 5cm and 5 degrees, formatting the results in a LaTeX table.

    Args:
        DATA_FOLDER (str, optional): Path to the dataset folder containing scene results. Defaults to "Data/Final_Dataset".
        prefixes (list, optional): List of prefixes representing different methods. Defaults to common methods.

    Returns:
        None: Prints the generated LaTeX table.
    """
    # Dictionary to store metrics for each prefix
    all_metrics = {prefix: {"translation_errors": [], "rotation_errors": [], "adds_scores": [], "accuracy_5cm_5deg": []} for prefix in prefixes}

    for prefix in prefixes:
        for scene in sorted(os.listdir(DATA_FOLDER)):
            scene_folder = os.path.join(DATA_FOLDER, scene)

            # Check if the JSON file for this prefix is available for each object
            if os.path.exists(os.path.join(scene_folder, prefix + "results.json")):
                json_file = os.path.join(scene_folder, prefix + "results.json")
                json_data = json.load(open(json_file))
                
                # Extract metrics
                translation_error = json_data['object_metrics']['rmse_position'] * 100  # Convert to cm
                rotation_error = json_data['object_metrics']['rmse_rotation'] * (180 / np.pi)  # Convert to degrees
                adds_score = json_data['object_metrics']['adds_score'] * 100  # Convert to percentage
                accuracy_5cm_5deg = json_data['object_metrics']['acc_5cm_5°'] * 100  # Convert to percentage

                # Append metrics for current prefix
                all_metrics[prefix]["translation_errors"].append(translation_error)
                all_metrics[prefix]["rotation_errors"].append(rotation_error)
                all_metrics[prefix]["adds_scores"].append(adds_score)
                all_metrics[prefix]["accuracy_5cm_5deg"].append(accuracy_5cm_5deg)
    
    # Prepare LaTeX content for each prefix
    latex_content = ""
    for prefix, metrics in all_metrics.items():
        # Calculate median and standard deviation for translation and rotation errors
        if metrics["translation_errors"]:
            median_translation = np.median(metrics["translation_errors"])
            std_translation = np.std(metrics["translation_errors"])

            median_rotation = np.median(metrics["rotation_errors"])
            std_rotation = np.std(metrics["rotation_errors"])

            # ADD-S score and accuracy calculations
            mean_adds_score = np.mean(metrics["adds_scores"])
            mean_accuracy = np.mean(metrics["accuracy_5cm_5deg"])

            # Percentile within standard deviation for translation and rotation errors
            within_std_dev_translation = [
                val for val in metrics["translation_errors"]
                if abs(val - median_translation) <= std_translation
            ]
            percentile_translation = (len(within_std_dev_translation) / len(metrics["translation_errors"])) * 100

            within_std_dev_rotation = [
                val for val in metrics["rotation_errors"]
                if abs(val - median_rotation) <= std_rotation
            ]
            percentile_rotation = (len(within_std_dev_rotation) / len(metrics["rotation_errors"])) * 100

            # Format for LaTeX with percentiles in parentheses
            adds_score_str = f"${mean_adds_score:.2f}\\%$"
            translation_str = f"${median_translation:.2f} \\pm {std_translation:.2f} \\;  ({percentile_translation:.1f}\\%)$"
            rotation_str = f"${median_rotation:.2f} \\pm {std_rotation:.2f} \\;  ({percentile_rotation:.1f}\\%)$"
            accuracy_str = f"${mean_accuracy:.2f}\\%$"

            # Add row for this prefix
            latex_content += f"{prefix} & {adds_score_str} & {translation_str} & {rotation_str} & {accuracy_str} \\\\\n"
        else:
            # Placeholder for missing data
            latex_content += f"{prefix} & $--$ & $--$ & $--$ & $--$ \\\\\n"

    # Output LaTeX formatted table content
    print(latex_content)


def ablation_study_table(DATA_FOLDER="Data/Final_Dataset", methods=["headpose_", "BundleTrack_", "FoundationPose_", "BundleSDF_", ""]):
    """Generate a LaTeX table for ablation study comparing full and pose-only configurations.

    This function extracts RMSE position errors, ADD scores, and ADD-S scores for different
    configurations of each method (pose-only vs full). The results are formatted into a
    LaTeX table, showing the impact of different model configurations.

    Args:
        DATA_FOLDER (str, optional): Path to the dataset folder containing scene results. Defaults to "Data/Final_Dataset".
        methods (list, optional): List of method prefixes. Defaults to common methods.

    Returns:
        None: Prints the generated LaTeX table.
    """
    # Initialize list to store LaTeX rows for each method configuration
    table_rows = []

    for method in methods:
        method_rows = []

        # Determine the configuration file names for "pose only" and "full" configurations
        if method == "headpose_":
            configs = {"full": "\\xmark"}
        else:
            configs = {"pose_only": "\\cmark", "full": "\\xmark"}

        for config_key, config_mark in configs.items():
            # Use "pose_only_" prefix only for pose-only configuration
            config_prefix = ("pose_only_" if config_key == "pose_only" else "") + method
            rmse_positions = []
            adds_scores = []
            add_s_scores = []

            # Search through all subfolders in DATA_FOLDER
            for scene in sorted(os.listdir(DATA_FOLDER)):
                scene_folder = os.path.join(DATA_FOLDER, scene)
                json_file_path = os.path.join(scene_folder, config_prefix + "results.json")
                
                # Check if the JSON file for this configuration exists in the current scene
                if os.path.exists(json_file_path):
                    with open(json_file_path) as json_file:
                        json_data = json.load(json_file)

                        # Append metrics to lists
                        rmse_positions.append(json_data['object_metrics']['rmse_position'] * 100)  # Convert to cm
                        adds_scores.append(json_data['object_metrics']['add_score'] * 100)
                        add_s_scores.append(json_data['object_metrics']['adds_score'] * 100)

            # Calculate the average for each metric if there are values
            avg_rmse_position = np.mean(rmse_positions) if rmse_positions else None
            avg_adds_score = np.mean(adds_scores) if adds_scores else None
            avg_add_s_score = np.mean(add_s_scores) if add_s_scores else None

            # Format values for LaTeX
            rmse_position_str = f"${avg_rmse_position:.2f}$" if avg_rmse_position is not None else "$--$"
            adds_score_str = f"${avg_adds_score:.2f}$" if avg_adds_score is not None else "$--$"
            add_s_score_str = f"${avg_add_s_score:.2f}$" if avg_add_s_score is not None else "$--$"
            method_rows.append(f"{config_mark} & {rmse_position_str} & {adds_score_str} & {add_s_score_str}")

        # Combine the configurations for this method into the LaTeX format
        if len(method_rows) == 1:  # For single-row methods like HeadPose
            table_rows.append(f"{method} & {method_rows[0]} \\\\\n")
        else:  # For multi-row methods like FoundationPose, BundleSDF, and Ours
            table_rows.append(f"\\multirow{{2}}{{*}}{{{method}}} & {method_rows[0]} \\\\\n")
            table_rows.append(f"& {method_rows[1]} \\\\\n")
        table_rows.append("\\midrule\n")
    
    # Construct the LaTeX table content
    latex_content = "\\begin{table}[t]\n"
    latex_content += "\\setlength{\\tabcolsep}{3pt} % Reduce the column spacing\n"
    latex_content += "\\renewcommand{\\arraystretch}{1.3} % Adjust row height for better vertical centering\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{\\textbf{\\textsc{Ablation Study}:} Evaluation of methods with and without pose-only configuration}\n"
    latex_content += "\\begin{tabular}{l c c c c}\n"  # Added an extra column for ADD-S
    latex_content += "\\toprule\n"
    latex_content += "Method & Pose Only & $\\textbf{T}_\\text{err}$ (cm) & ADD-0.1d & ADD-S \\\\\n"
    latex_content += "\\midrule\n"
    latex_content += "".join(table_rows)
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\label{tab:ablation_study}\n"
    latex_content += "\\end{table}\n"
    
    print(latex_content)



if __name__ == "__main__":
    pass