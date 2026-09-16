import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def inspect_dataset():
    print("========================================")
    print("MEDFINDER AI - Dataset Inspector")
    print("========================================")
    
    dataset_root = os.getenv("CHEXPERT_ROOT", ".")
    print(f"Configured Dataset Root: {dataset_root}")
    
    train_csv_path = os.path.join(dataset_root, "train.csv")
    valid_csv_path = os.path.join(dataset_root, "valid.csv")
    
    if not os.path.exists(train_csv_path):
        print(f"[ERROR] train.csv not found at {train_csv_path}")
        return
        
    print(f"[SUCCESS] Found train.csv at {train_csv_path}")
    
    if os.path.exists(valid_csv_path):
        print(f"[SUCCESS] Found valid.csv at {valid_csv_path}")
    else:
        print(f"[WARNING] valid.csv not found at {valid_csv_path}")
        
    print("\nLoading training data...")
    try:
        train_df = pd.read_csv(train_csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load train.csv: {e}")
        return
        
    print(f"\nDataset Size: {len(train_df)} rows")
    
    print("\nColumns:")
    for col in train_df.columns:
        print(f" - {col}")
        
    print("\nSample Rows:")
    print(train_df.head(2).to_string())
    
    print("\nChecking for missing files...")
    # Path is usually 'CheXpert-v1.0-small/train/...'
    # Wait, the path in CSV might include 'CheXpert-v1.0-small/'.
    # If our dataset_root is the directory containing train.csv,
    # the image path in the CSV might be relative to dataset_root, or its parent.
    # Let's check the first path.
    missing_count = 0
    checked_count = 0
    max_check = 1000 # Only check first 1000 to save time
    
    for idx, row in train_df.head(max_check).iterrows():
        img_path_in_csv = row['Path']
        # Depending on structure, image path might be relative to the dataset root or its parent.
        # Often the CSV contains 'CheXpert-v1.0-small/train/...', 
        # and if the dataset_root is 'CheXpert-v1.0-small', we need to strip it or handle it.
        # Let's just try to resolve it.
        
        full_img_path = os.path.join(dataset_root, img_path_in_csv)
        
        # If the root already ends with 'CheXpert-v1.0-small' and the path starts with it, they might duplicate.
        if img_path_in_csv.startswith("CheXpert-v1.0-small/") and dataset_root.endswith("CheXpert-v1.0-small"):
            # strip it
            full_img_path = os.path.join(dataset_root, img_path_in_csv.replace("CheXpert-v1.0-small/", ""))
        # Or maybe it's just relative to the parent of dataset_root
        elif not os.path.exists(full_img_path):
            parent_dir = os.path.dirname(dataset_root)
            alt_path = os.path.join(parent_dir, img_path_in_csv)
            if os.path.exists(alt_path):
                full_img_path = alt_path
                
        if not os.path.exists(full_img_path):
            missing_count += 1
        checked_count += 1
            
    print(f"Checked {checked_count} image paths.")
    if missing_count > 0:
        print(f"[WARNING] Found {missing_count} missing images out of {checked_count} checked.")
    else:
        print(f"[SUCCESS] All {checked_count} checked image paths exist.")
        
    print("\nBasic Label Statistics (Positive Cases):")
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    for label in labels:
        if label in train_df.columns:
            positive_count = (train_df[label] == 1.0).sum()
            print(f" - {label}: {positive_count} cases")
            
    print("\nInspection Complete.")

if __name__ == "__main__":
    inspect_dataset()
