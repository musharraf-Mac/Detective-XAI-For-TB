import pandas as pd
import os
import random

# Derive absolute path to data folder relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data_pr/TB_Data"))
SPLITS = ["train", "test", "val"]
CXR_CLASSES = ["normal", "TB", "abnormal"]
OOD_CLASS = "unknown"

random.seed(42)  # For reproducibility

SAMPLE_RATE = 0.1  # 10% of the data will be sampled

def get_image_paths(folder):
    """Get all image file paths from a folder."""
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
# prepare data

def create_model_a_csv(split):
    """Create a CSV for CXR detection model A."""
    records=[]
    split_dir = os.path.join(BASE_DIR, split)
    
    # 1. Sample 10% of chest X-ray classes (label = 1)
    for cls in CXR_CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        all_images = get_image_paths(cls_dir)
        
        # Sample 10%
        sample_size = max(1, int(len(all_images) * SAMPLE_RATE))
        sampled = random.sample(all_images, sample_size)
        
        for img_path in sampled:
            records.append({
                "filepath": img_path,
                "label": 1,
                "class_name": "chest_xray"
            })
        
        print(f"  {split}/{cls}: {len(all_images)} total → {sample_size} sampled (10%)")
    
    # 2. Use ALL unknown images (label = 0)
    unknown_dir = os.path.join(split_dir, OOD_CLASS)
    unknown_images = get_image_paths(unknown_dir)
    
    for img_path in unknown_images:
        records.append({
            "filepath": img_path,
            "label": 0,
            "class_name": "not_chest_xray"
        })
    
    print(f"  {split}/unknown: {len(unknown_images)} total → {len(unknown_images)} used (100%)")
    
    # 3. Shuffle and create DataFrame
    random.shuffle(records)
    df = pd.DataFrame(records)
    
    # 4. Save
    output_path = f"model_a_{split}.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ {output_path} created!")
    print(f"   Total: {len(df)}")
    print(f"   Chest X-ray: {sum(df['label'] == 1)}")
    print(f"   Not Chest X-ray: {sum(df['label'] == 0)}")
    print()
    
    return df

print("=" * 60)
print("CREATING MODEL A CSVs (Chest X-ray Detector)")
print("=" * 60)

for split in SPLITS:
    print(f"\n--- {split.upper()} ---")
    create_model_a_csv(split)

    