from pathlib import Path

# Define the folder paths and the prefix to add
tasks = [
    {
        "folder": r"C:\Users\MSI\Documents\Education\BSc.IT\Semester_6\Intelligence system\Research\Detective-XAI-For-TB\data_pr\TB_Data\Abnormal_Other\Pne_1",
        "prefix": "pn"
    },
    {
        "folder": r"C:\Users\MSI\Documents\Education\BSc.IT\Semester_6\Intelligence system\Research\Detective-XAI-For-TB\data_pr\TB_Data\Abnormal_Other\Pne_11",
        "prefix": "pn1"
    },
    {
        "folder": r"C:\Users\MSI\Documents\Education\BSc.IT\Semester_6\Intelligence system\Research\Detective-XAI-For-TB\data_pr\TB_Data\Abnormal_Other\Pneu_ch2",
        "prefix": "pc2"
    },
    {
        "folder": r"C:\Users\MSI\Documents\Education\BSc.IT\Semester_6\Intelligence system\Research\Detective-XAI-For-TB\data_pr\TB_Data\Abnormal_Other\Pneu_CXR",
        "prefix": "pcx"
    },
]

for task in tasks:
    folder = Path(task["folder"])
    prefix = task["prefix"]

    if not folder.exists():
        print(f"Skipping: Folder {folder} does not exist.")
        continue

    print(f"Processing folder: {folder} with prefix '{prefix}'...")
    count = 0
    for file in folder.iterdir():
        if file.is_file():
            # Check if file already starts with the prefix to avoid double-renaming
            if not file.name.startswith(f"{prefix}_"):
                file.rename(file.parent / f"{prefix}_{file.name}")
                count += 1
    print(f"Successfully renamed {count} files.")

print("All batch renaming tasks completed.")
