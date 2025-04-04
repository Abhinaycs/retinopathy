import os
import shutil
import pandas as pd

# Load CSV file
train_csv = "B:/train.csv"
df = pd.read_csv(train_csv)

# Check available columns
print("Available columns:", df.columns)

# Ensure the correct column names are used
if "diagnosis" not in df.columns:
    raise ValueError("Column 'diagnosis' not found! Check your CSV format.")

# Define paths
train_images_path = "B:/train_images/"
output_path = "B:/organized_train_images/"

# Create class directories
classes = df["diagnosis"].unique()
for c in classes:
    os.makedirs(os.path.join(output_path, str(c)), exist_ok=True)

# Move images into class folders
for _, row in df.iterrows():
    img_name = row["id_code"] + ".png"  # Assuming images are in PNG format
    img_path = os.path.join(train_images_path, img_name)
    dest_path = os.path.join(output_path, str(row["diagnosis"]), img_name)

    if os.path.exists(img_path):
        shutil.move(img_path, dest_path)
    else:
        print(f"Warning: {img_path} not found!")

print("Dataset successfully organized!")
