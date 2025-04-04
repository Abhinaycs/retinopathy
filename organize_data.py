import os
import shutil
import pandas as pd
import argparse
import glob

def organize_data(train_csv_path, test_csv_path, image_folder, output_folder, test_size=0.2):
    """
    Organizes dataset images into structured train/valid folders based on labels.

    Args:
    - train_csv_path: Path to training CSV file
    - test_csv_path: Path to testing CSV file
    - image_folder: Path where train images are stored
    - output_folder: Destination path for organized images
    - test_size: Proportion of data for validation set
    """

    # Create output directories
    train_output = os.path.join(output_folder, "train_images")
    valid_output = os.path.join(output_folder, "valid_images")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(valid_output, exist_ok=True)

    # Load CSV files
    train_df = pd.read_csv(train_csv_path)
    
    # Function to move images to respective folders
    def move_images(df, dest_folder):
        for _, row in df.iterrows():
            img_name = row['image'] + ".png"  # Ensure correct file format
            
            # Search for the image inside all subfolders
            matching_files = glob.glob(os.path.join(image_folder, "**", img_name), recursive=True)

            if matching_files:
                src_path = matching_files[0]  # Take the first match
                dest_path = os.path.join(dest_folder, row['label'], img_name)

                # Ensure label folder exists
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Move the image
                shutil.copy(src_path, dest_path)
            else:
                print(f"⚠️ Warning: {img_name} not found, skipping.")

    # Move train images
    print("📂 Organizing train images...")
    move_images(train_df, train_output)

    print("✅ Dataset organization completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize images into labeled folders")
    parser.add_argument("train_csv_path", type=str, help="Path to training CSV file")
    parser.add_argument("test_csv_path", type=str, help="Path to testing CSV file")
    parser.add_argument("image_folder", type=str, help="Path to train images folder")
    parser.add_argument("output_folder", type=str, help="Path to output dataset folder")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation set size (default: 0.2)")

    args = parser.parse_args()
    organize_data(args.train_csv_path, args.test_csv_path, args.image_folder, args.output_folder, args.test_size)
