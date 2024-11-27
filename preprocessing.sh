#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/workspace"
    exit 1
fi

WORKSPACE_DIR="$1"

cd "$WORKSPACE_DIR" || { echo "Workspace directory not found!"; exit 1; }


read -p "Enter username: " username
read -sp "Enter password: " password
echo

aria_mps single -i . --no-ui -u "$username" -p "$password"


for vrs_file in *.vrs; do
    base_name="${vrs_file%.vrs}"

    json_file="${base_name}.vrs.json"
    if [[ -f "$json_file" ]]; then
        mkdir -p "$base_name"

        mv "$vrs_file" "$json_file" "$base_name/"
        echo "Moved $vrs_file and $json_file to $base_name/"

        cd "$base_name" || { echo "Failed to enter $base_name"; exit 1; }

        # Move the mps_base_name_vrs subfolder to the base_name subfolder
        mps_subfolder="../mps_${base_name}_vrs"
        if [[ -d "$mps_subfolder" ]]; then
            mv "$mps_subfolder" .
            echo "Moved $mps_subfolder to $base_name/"
        else
            echo "Warning: No mps_${base_name}_vrs subfolder found"
        fi

        cd ..
    else
        echo "Warning: No matching .vrs.json found for $base_name"
    fi
done