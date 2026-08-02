import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

RAW_DIR = os.path.join("..","data_pr", "Raw")
PROCESSED_DIR = os.path.join("..","data_pr", "preprocessed")

print("Setup complete")
print(f"Raw data directory: {os.listdir(RAW_DIR)}")
                        
ch_img_dir = os.path.join(RAW_DIR,"Chinaset_AllFiles", "CXR_png" )
ch_txt_dir = os.path.join(RAW_DIR,"Chinaset_AllFiles", "ClinicalReadings" )

sample_file = os.listdir(ch_txt_dir)[0]
with open(os.path.join(ch_txt_dir, sample_file)) as f:
    print(f"File name: {sample_file}")
    print(f"Content: \n{f.read()}")
    
def parse_clinical_text(text):
    text = text.lower()
    age_match = re.search(r"(\d{1,3})\s*y", text)
    age = int(age_match.group(1)) if age_match else None
    if "female" in text:
        sex = "F"
    elif "male" in text:
        sex = "M"
    else:
        sex = None
    return age, sex

records = []
for fname in sorted(os.listdir(ch_txt_dir)):
    if not fname.endswith(".txt"):
        continue
    image_id = fname.split(".")[0]
    label = int(re.search(r"_(\d)\.txt$", fname).group(1))
    
    with open(os.path.join(ch_txt_dir, fname), errors="ignore") as f:
        raw_text = f.read()
    
    age, sex = parse_clinical_text(raw_text)
    
    records.append({
        "image_id": image_id,
        "filepath": os.path.join(ch_img_dir, image_id + ".png"),
        "label": label,
        "age": age,
        "sex": sex
    })

df = pd.DataFrame(records)
df.head(10)
df.tail(10)

print("Label distribution:")
print(df['label'].value_counts())
print(f"\nMissing sex:{df['sex'].isna().sum()}/ {len(df)}")
print(f"Missing age:{df['age'].isna().sum()}/ {len(df)}")

df['label'].value_counts().plot(kind='bar', title ='Class Distribution (0 = Normal, 1 = TB)')
plt.show()

fig, axes = plt.subplots(2,3, figsize = (12,8))

for i, ax in enumerate(axes.flat):
    row = df.iloc[i]
    img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE) # Read image in grayscale
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {row['label']} | Age: {row['age']} | Sex: {row['sex']}")
    ax.axis('off')
    
output_path = os.path.join(PROCESSED_DIR, "ch_metadata.csv")
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} records to {output_path}")
