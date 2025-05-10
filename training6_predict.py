
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder

CONFIG = {
    "batch_size": 10,
    "target_size": (320, 480),  # width, height
    "best_model": "training6/training6_best_model.keras",
    "label_encoder_path": "training6/training6_label_encoder.npy"
}

class TestDataGenerator(Sequence):
    """Handles loading and preprocessing of test images"""
    def __init__(self, image_dir, batch_size=CONFIG['batch_size'], target_size=CONFIG['target_size']):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.batch_size = batch_size
        self.target_size = target_size
        
    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_files = self.image_files[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((len(batch_files), self.target_size[1], self.target_size[0], 3), dtype=np.float32)
        
        for i, filename in enumerate(batch_files):
            img_path = os.path.join(self.image_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Force 3-channel read
            if img is None:
                raise ValueError(f"Could not read image at {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            X[i] = img / 255.0  # Normalize
            
        return X
    
    def get_image_ids(self):
        """Extracts clean image IDs from filenames"""
        return [f.split('_')[0] + '.jpg' for f in self.image_files]

def load_assets():
    """Loads model and label encoder"""
    with open(CONFIG['label_encoder_path'], 'rb') as f:
        classes = np.load(f, allow_pickle=True)
    
    le = LabelEncoder()
    le.classes_ = classes
    model = tf.keras.models.load_model(CONFIG['best_model'])
    return model, le

def predict_and_save(test_image_dir, output_csv="predictions.csv"):
    """Main prediction workflow"""
    model, le = load_assets()
    test_gen = TestDataGenerator(test_image_dir)
    
    # Generate predictions
    predictions = []
    for batch in test_gen:
        preds = model.predict(batch, verbose=0).argmax(axis=1)
        predictions.extend(preds)
    
    # Create and save results
    results = pd.DataFrame({
        'image_id': test_gen.get_image_ids(),
        'label': le.inverse_transform(predictions)
    })
    results.to_csv(output_csv, index=False)
    print(f"Success! Predictions saved to {output_csv}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="Dataset/preprocessed_test_spectral_images", help="Path to test images")
    parser.add_argument("--output", default="training6/predictions.csv", help="Output CSV path")
    args = parser.parse_args()
    
    predict_and_save(args.test_dir, args.output)