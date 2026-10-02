from pathlib import Path

folder_path = input("Enter the folder path: ")
folder = Path(folder_path)
new_name=input("Enter the letter to add to the file name: ")

if not folder.exists():
    print(f"The folder {folder} does not exist.")
    exit()
else:
    for file in folder.iterdir():
        if file.is_file():            
            file.rename(file.parent / f"{new_name}_{file.name}")
            
print("Renaming complete.")