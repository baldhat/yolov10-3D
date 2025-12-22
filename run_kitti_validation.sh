#!/bin/bash

# --- Configuration ---
KITTI_ROOT="/storage/group/deepscenario/KITTI/kitti_raw_data"
KITTI_CONFIG_FILE="/usr/wiss/mejo/Development/yolov10-3D/ultralytics/data/datasets/kitti.py"
VIDEO_CONFIG_FILE="/usr/wiss/mejo/Development/yolov10-3D/ultralytics/utils/create_video.py"
VALIDATION_COMMAND="python ultralytics/cfg/__init__.py val model=/usr/wiss/mejo/storage-deepscenario/jonathan_for_johannes/yolov10-3Dx_2/weights/best.pt data=kitti.yaml save_dir=/usr/wiss/mejo/storage/user/_archiv_paper/2026_CVPR_LeAD-M3D/von_johannes/kitti_vis batch=8"
DIR1_OPTIONS=("2011_09_26" "2011_09_28" "2011_09_29" "2011_09_30" "2011_10_03")

# Loop through each DIR1 (date folder)
for DIR1 in "${DIR1_OPTIONS[@]}"; do
    
    CURRENT_DIR1_PATH="${KITTI_ROOT}/${DIR1}"
    echo "--- Processing DIR1: ${DIR1} ---"
    
    DIR2_OPTIONS=$(find "$CURRENT_DIR1_PATH" -maxdepth 1 -type d -name "*sync" -exec basename {} \;)

    # Loop through each DIR2 (drive folder) found
    for DIR2 in $DIR2_OPTIONS; do
        
        echo "--> Processing DIR2: ${DIR2}"
        
        # --- Path and File Name Construction ---
        NEW_IMAGE_DIR="${KITTI_ROOT}/${DIR1}/${DIR2}/image_02/data/"
        NEW_PNG_NAME="${DIR1}_${DIR2}_rgb.mp4"
        NEW_SVG_NAME="${DIR1}_${DIR2}_bev.mp4"

        # 1. Update kitti.py (Line 66)
        echo "    Updating ${KITTI_CONFIG_FILE}..."
        sed -i "s#^.*self.image_dir = \".*kitti_raw_data/.*\"#        self.image_dir = \"${NEW_IMAGE_DIR}\"#" "$KITTI_CONFIG_FILE"
        
        # 3. Run the validation command
        echo "    Executing validation command..."
        RANDOM_SEED=$RANDOM
        $VALIDATION_COMMAND seed=$RANDOM_SEED
        
        echo "    Validation for ${DIR1}/${DIR2} complete."
        echo "" 
        
    done
    
    echo "--- All DIR2 complete for ${DIR1} ---"
    echo ""

done

echo "✅ All KITTI date and drive folders processed successfully."