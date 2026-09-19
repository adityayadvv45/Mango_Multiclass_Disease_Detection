import os
import shutil
import random
import csv
from glob import glob

def create_splits(source_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    # 1. Get all images
    images = glob(os.path.join(source_dir, "*.jpg"))
    if not images:
        print("No images found to split.")
        return
        
    print(f"Found {len(images)} images.")
    
    # 2. Extract labels and prepare data
    data = []
    # Possible classes based on the base dataset
    all_possible_classes = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", 
                            "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould", "uploads"]
    
    for img_path in images:
        filename = os.path.basename(img_path)
        # Format: synthetic_Class1_Class2_ID.jpg
        parts = filename.replace('.jpg', '').split('_')
        
        # parts[0] = 'synthetic'
        # parts[1] = Class1
        # parts[2] = Class2
        # parts[3] = ID
        
        if len(parts) >= 4:
            class1 = parts[1]
            class2 = parts[2]
            
            # Create a multi-hot encoding or simply a list of labels
            # For simplicity, let's make a dictionary of {class_name: 1 or 0}
            labels = {cls: 0 for cls in all_possible_classes}
            if class1 in labels:
                labels[class1] = 1
            if class2 in labels:
                labels[class2] = 1
                
            data.append({
                "filename": filename,
                "filepath": img_path,
                "labels": labels
            })
            
    # 3. Shuffle data
    random.seed(42) # For reproducibility
    random.shuffle(data)
    
    # 4. Calculate split indices
    train_end = int(len(data) * split_ratio[0])
    val_end = train_end + int(len(data) * split_ratio[1])
    
    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }
    
    # 5. Create folders and copy files
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        print(f"Processing {split_name} split with {len(split_data)} images...")
        
        split_img_dir = os.path.join(output_dir, split_name, "images")
        os.makedirs(split_img_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            # Header: filename, Anthracnose, Bacterial Canker, ...
            fieldnames = ['filename'] + all_possible_classes
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in split_data:
                # Copy image
                dest_path = os.path.join(split_img_dir, item["filename"])
                shutil.copy2(item["filepath"], dest_path)
                
                # Write CSV row
                row = {'filename': item["filename"]}
                row.update(item["labels"])
                writer.writerow(row)
                
    print(f"Splits created successfully in {output_dir}")

if __name__ == "__main__":
    source_directory = "/Users/abhinavsahu/Downloads/projects/mangoDisease/synthetic_multi_disease"
    output_directory = "/Users/abhinavsahu/Downloads/projects/mangoDisease/dataset_splits"
    
    create_splits(source_directory, output_directory)
