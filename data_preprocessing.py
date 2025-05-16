import pandas as pd
import numpy as np
from cleanvision import Imagelab
import os
import pickle

def preprocess_data():
    # Load and clean data
    df = pd.read_csv("Dataset/meta_train.csv")
    print("Initial shape of the dataset: ", df.shape)

    # Run CleanVision analysis
    imagelab = Imagelab(data_path="Dataset/train_images")
    imagelab.find_issues()

    # Handle duplicates
    unreliable_sets = []
    
    for set_type in ["exact_duplicates", "near_duplicates"]:
        for img_set in imagelab.info[set_type]["sets"]:
            # Get image filenames without paths
            img_names = [os.path.basename(img) for img in img_set]
            
            # Compare metadata for all pairs in the set
            for i in range(len(img_names)):
                for j in range(i+1, len(img_names)):
                    img1 = img_names[i]
                    img2 = img_names[j]
                    
                    # Get metadata (excluding image_id)
                    meta1 = df[df["image_id"] == img1].drop("image_id", axis=1).reset_index(drop=True)
                    meta2 = df[df["image_id"] == img2].drop("image_id", axis=1).reset_index(drop=True)
                    
                    # Compare metadata after ensuring same columns and indices
                    if not meta1.equals(meta2):
                        print(f"Metadata mismatch in {set_type}: {img1} vs {img2}")
                        unreliable_sets.append(img_set)
                        break

    # Remove duplicates (keep first occurrence in each set)
    for set_type in ["exact_duplicates", "near_duplicates"]:
        for img_set in imagelab.info[set_type]["sets"]:
            img_names = [os.path.basename(img) for img in img_set]
            df = df[~df["image_id"].isin(img_names[1:])]  # Keep first, remove others

    # Remove images from unreliable sets
    for img_set in unreliable_sets:
        img_names = [os.path.basename(img) for img in img_set]
        df = df[~df["image_id"].isin(img_names)]
    
    print("Final shape of the dataset: ", df.shape)
    return df

def save_processed_data(df, save_path=""):
    """Save processed data for later use"""
    if not save_path:
        save_path = "processed_data"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(os.path.join(save_path, "cleaned_metadata.csv"), index=False)
    
    # Save as pickle for faster loading
    with open(os.path.join(save_path, "cleaned_data.pkl"), "wb") as f:
        pickle.dump(df, f)
    
    print(f"Data saved to {save_path}/")

if __name__ == "__main__":
    cleaned_df = preprocess_data()
    save_processed_data(cleaned_df, save_path="")  # Empty string uses default path