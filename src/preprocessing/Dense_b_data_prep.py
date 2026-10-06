import os
import pandas as pd
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..","..","data_pr","TB_Data"))
OUTPUT_CSV = "model_b_dataset.csv"
OUTPUT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..","..","data_pr","preprocessed",OUTPUT_CSV))

CLASSES = {"normal": 0, "tb": 1, "abnormal": 2}

SPILITS = ["train", "test", "val"]

random.seed(42)  # For reproducibility

# Getting all images
def get_image_paths(folder):
    """Get all image file paths from a folder."""
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

# Build dataset for model B
records= []

for split in SPILITS:
    split_dir= os.path.join(BASE_DIR, split)
    
    if not os.path.exists(split_dir):
        print(f"Warning: Split directory '{split_dir}' does not exist. Skipping.")
        continue
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(split_dir, class_name)
        image_paths = get_image_paths(class_dir)
        
        for img_path in image_paths:
            records.append({
                "filepath": img_path,
                "label": label,
                "class_name": class_name,
                "split": split
            })
        
        print(f"  {split}/{class_name}: {len(image_paths)} images found.")
        
# Create DataFrame and save to CSV
df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset for model B saved to {OUTPUT_PATH}. Total records: {len(df)}")


