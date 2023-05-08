import os
import zipfile

# Get the directory path of the current file
dir_path = os.path.dirname(os.path.abspath(__file__))

# Set the path and name of the output zip
zip_path = "assignment_3_submission.zip"

# Check if the output zip file already exists
if os.path.exists(zip_path):
    # If the file exists, delete it
    os.remove(zip_path)
    print(f"Deleted existing {zip_path}")

# Open the output zip file in write mode
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

    # Walk through all the directories and files in the folder
    for foldername, subfolders, filenames in os.walk(dir_path):
        
        # Remove subfolders to ignore
        subfolders[:] = [subfolder for subfolder in subfolders 
                         if subfolder != ".git"]
        
        # Loop through all the files in the current directory
        for filename in filenames:
            
            # Only include .py files and files in the visualization folder
            if (not filename.endswith('.zip')) and (not filename.endswith('.gitignore')) and \
                (not foldername.endswith('.git')) and (not foldername.endswith('.zip')) and \
                (not foldername.endswith('.gitignore')) :
                
                # Get the full path of the file
                file_path = os.path.join(foldername, filename)
                
                # Add the file to the zip archive
                zipf.write(file_path, arcname=file_path[len(dir_path)+1:])
                
# Print a message when the zip operation is complete
print(f"All files in {dir_path} and its subdirectories have been zipped to {zip_path}")
