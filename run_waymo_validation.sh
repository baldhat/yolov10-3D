#!/bin/bash

# --- Configuration ---
# Root directory where the Waymo segment folders are located (your DIR1 options)
WAYMO_ROOT="/usr/wiss/mejo/storage-deepscenario/waymo/validation_org"

# Configuration file to be updated
WAYMO_CONFIG_FILE="/usr/wiss/mejo/Development/yolov10-3D/ultralytics/data/datasets/waymo.py"

# Validation command (modified to remove the 'data=kitti.yaml' part, assuming waymo.yaml is used or implied)
# NOTE: Ensure you update 'data=waymo.yaml' if it's explicitly needed, or check where it's configured.
# Assuming the model path and save_dir are correct.
VALIDATION_COMMAND="python ultralytics/cfg/__init__.py val model=/usr/wiss/mejo/storage/user/_archiv_paper/2026_CVPR_LeAD-M3D/von_johannes/yolov10-3D_waymo_ours_x_lr.0025_dalw2-0.35/weights/best.pt data=waymo.yaml save_dir=/usr/wiss/mejo/storage/user/_archiv_paper/2026_CVPR_LeAD-M3D/von_johannes/kitti_vis batch=8"


# 1. Determine all DIR1 options (segment folders) dynamically
# Find all directories in WAYMO_ROOT and extract their names
echo "--- Determining Waymo Segment Directories ---"
DIR1_OPTIONS=$(find "$WAYMO_ROOT" -maxdepth 1 -type d -not -name "$(basename "$WAYMO_ROOT")" -exec basename {} \;)

# Loop through each DIR1 (segment folder)
for DIR1 in $DIR1_OPTIONS; do
    
    echo "--- Processing Segment (DIR1): ${DIR1} ---"
    
    # --- Variable Construction ---
    # The new line to be inserted into waymo.py, including required Python indentation
    # NOTE: We use single quotes for the string value inside the Python variable assignment.
    NEW_SEGMENT_LINE="        segment = '${DIR1}'"
    
    # 1. Update waymo.py (Segment Variable)
    # The 'sed' command searches for the segment variable assignment and replaces the ENTIRE line.
    # We use '#' as the delimiter.
    echo "    Updating ${WAYMO_CONFIG_FILE}: setting segment = '${DIR1}'"
    sed -i "s|^.*segment = .*$|$NEW_SEGMENT_LINE|" "$WAYMO_CONFIG_FILE"
    
    # 2. Run the validation command
    echo "    Executing validation command..."
    RANDOM_SEED=$RANDOM
    $VALIDATION_COMMAND seed=$RANDOM_SEED
    
    echo "    Validation for segment ${DIR1} complete."
    echo "" 

done

echo "✅ All Waymo segments processed successfully."