import os
import cv2
import numpy as np
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

def blend_kitti_validation_data(base_dir, output_dir, num_samples):
    # Standard KITTI paths inside the source folder
    img_dir = os.path.join(base_dir, 'image_2')
    lbl_dir = os.path.join(base_dir, 'label_2')
    cal_dir = os.path.join(base_dir, 'calib')
    val_list_path = os.path.join(base_dir, 'ImageSets', 'val.txt')

    # Output paths
    out_img_dir = os.path.join(output_dir, 'image_2')
    out_lbl_dir = os.path.join(output_dir, 'label_2')
    out_cal_dir = os.path.join(output_dir, 'calib')
    out_set_dir = os.path.join(output_dir, '../ImageSets')

    for d in [out_img_dir, out_lbl_dir, out_cal_dir, out_set_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load the validation IDs to ensure we only use the val split
    if not os.path.exists(val_list_path):
        print(f"Error: Could not find val.txt at {val_list_path}")
        return

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Group validation images by identical calibration parameters
    print("Indexing calibration files for validation set...")
    calib_groups = defaultdict(list)
    
    for s_id in val_ids:
        cal_path = os.path.join(cal_dir, f"{s_id}.txt")
        if not os.path.exists(cal_path):
            continue
            
        with open(cal_path, 'r') as f:
            cal_content = f.read().strip()
            calib_groups[cal_content].append(s_id)

    # Filter groups that have at least 2 images
    valid_groups = [ids for ids in calib_groups.values() if len(ids) >= 2]

    if not valid_groups:
        print("Error: No validation images share the same calibration.")
        return

    generated_ids = []

    # 3. Generation Loop
    print(f"Generating {num_samples} blended samples...")
    for i in tqdm(range(num_samples)):
        # Pick a group and two random images from it
        group = random.choice(valid_groups)
        id1, id2 = random.sample(group, 2)

        # Create new ID based on the loop index (standard KITTI 6-digit format)
        new_id = f"{i:06d}"
        generated_ids.append(new_id)

        # Image processing
        img1 = cv2.imread(os.path.join(img_dir, f"{id1}.png"))
        img2 = cv2.imread(os.path.join(img_dir, f"{id2}.png"))

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Blending: I_new = 0.5 * I_1 + 0.5 * I_2
        blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        # Label processing
        with open(os.path.join(lbl_dir, f"{id1}.txt"), 'r') as f:
            lines1 = f.readlines()
        with open(os.path.join(lbl_dir, f"{id2}.txt"), 'r') as f:
            lines2 = f.readlines()

        # Calibration data (identical for both, so we just take one)
        with open(os.path.join(cal_dir, f"{id1}.txt"), 'r') as f:
            calib_data = f.read()

        # Save files
        cv2.imwrite(os.path.join(out_img_dir, f"{new_id}.png"), blended_img)
        
        with open(os.path.join(out_lbl_dir, f"{new_id}.txt"), 'w') as f:
            f.writelines(lines1 + lines2)
            
        with open(os.path.join(out_cal_dir, f"{new_id}.txt"), 'w') as f:
            f.write(calib_data)

    # 4. Create the new val.txt file
    with open(os.path.join(out_set_dir, 'val.txt'), 'w') as f:
        for gid in generated_ids:
            f.write(f"{gid}\n")

    print(f"Success! New dataset and ImageSets/val.txt created at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blended KITTI dataset from validation split.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to original KITTI training folder")
    parser.add_argument("--output", type=str, required=True, help="Target path for new dataset")
    parser.add_argument("--n", type=int, default=10, help="Number of images to generate")
    
    args = parser.parse_args()
    blend_kitti_validation_data(args.base_dir, args.output, args.n)