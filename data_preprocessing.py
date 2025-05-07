import tensorflow as tf
import numpy as np
import os
import hashlib
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from enum import Enum
from numpy.typing import NDArray
from typing import TypeAlias

# --- Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BUFFER_SIZE = tf.data.AUTOTUNE # Dynamic buffer size

# Define class names based on the subdirectories in Dataset/new_preprocessed_images/
CLASS_NAMES = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]
NUM_CLASSES = len(CLASS_NAMES)

# Define type aliases for clarity
MatLike: TypeAlias = NDArray[np.uint8]

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

# --- Duplicate Removal ---
def calculate_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(4096) # Read in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def find_and_remove_duplicates(data_dir):
    """Finds and optionally removes duplicate images based on SHA-256 hash."""
    hashes = {}
    duplicates = []
    all_files = []
    print("Scanning for duplicates...")
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    file_path = img_path
                    try:
                        file_hash = calculate_sha256(file_path)
                        if file_hash in hashes:
                            print(f"Duplicate found: {file_path} (same as {hashes[file_hash]})")
                            duplicates.append(file_path)
                            # Optional: Remove duplicate file
                            # os.remove(file_path)
                            # print(f"Removed duplicate: {file_path}")
                        else:
                            hashes[file_hash] = file_path
                            all_files.append(file_path)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
    print(f"Scan complete. Found {len(duplicates)} duplicates.")
    unique_files = list(hashes.values())
    print(f"Total unique files found: {len(unique_files)}")
    return unique_files

# --- Image Preprocessing Functions ---

# Function to apply Gaussian Blur using OpenCV (expects NumPy array)
def apply_gaussian_blur_cv(image_np: MatLike, ksize=(5, 5), sigmaX=0) -> MatLike:
    """Applies Gaussian Blur to a NumPy image array."""
    blurred_image = cv2.GaussianBlur(image_np, ksize, sigmaX)
    return blurred_image

# Function to apply CLAHE using OpenCV (expects NumPy array)
def apply_clahe_cv(image_np: MatLike, clipLimit=2.0, tileGridSize=(8, 8)) -> MatLike:
    """Applies CLAHE to a NumPy image array (expects BGR format)."""
    # Convert to LAB color space if it's a color image
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Assuming input from TF is RGB, convert to BGR for OpenCV processing if needed
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl_channel = clahe.apply(l_channel)
        merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
        clahe_bgr_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        # Convert back to RGB for TensorFlow
        clahe_image = cv2.cvtColor(clahe_bgr_image, cv2.COLOR_BGR2RGB)
    else: # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        clahe_image = clahe.apply(image_np)
        # If grayscale, ensure it has 3 channels if TF expects it
        if len(clahe_image.shape) == 2:
             clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)

    return clahe_image

# Combined TensorFlow preprocessing function
def preprocess_image_tf(image):
    """Applies preprocessing steps using TensorFlow operations."""
    # Resize
    image = tf.image.resize(image, IMG_SIZE)

    # Gaussian Blur (using tf.numpy_function to wrap OpenCV)
    image = tf.numpy_function(
        func=apply_gaussian_blur_cv, 
        inp=[image], 
        Tout=tf.float32 # Ensure the output type matches expectations
    )
    # Set shape explicitly after numpy_function, as it might lose shape info
    image.set_shape(IMG_SIZE + (3,))

    # Normalization (already done by ImageDataGenerator or can be done here)
    # image = image / 255.0 # Example normalization

    return image

# --- Augmentation Functions ---
@tf.function
def augment_image(image):
    """Applies random augmentations."""
    # Random Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # HSV Jitter (+/- 20% saturation/value)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Rotation (+/- 15 degrees) - Requires tensorflow-addons
    try:
        import tensorflow_addons as tfa
        # Convert degrees to radians
        deg_to_rad = tf.constant(np.pi / 180.0, dtype=tf.float32)
        angle = tf.random.uniform([], minval=-15.0, maxval=15.0) * deg_to_rad
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    except ImportError:
        # Fallback if tfa is not installed - maybe just 90 degree rotations?
        # Or print a warning
        tf.print("Warning: tensorflow-addons not installed. Skipping rotation augmentation.")
        pass # Silently skip rotation if tfa not available

    # Ensure values are clipped [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# --- DataLoader Class ---
class DataLoader:
    # Update __init__ to use the globally defined CLASS_NAMES and NUM_CLASSES
    def __init__(self, data_dir, img_size=IMG_SIZE, batch_size=32, val_split=0.2, seed=42):
        self.data_dir = data_dir # Expecting path like './Dataset/new_preprocessed_images'
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = CLASS_NAMES # Use global list
        self.num_classes = NUM_CLASSES # Use global count
        self.val_split = val_split
        self.seed = seed
        self._prepare_datasets()

    def _get_paths_and_labels(self):
        image_paths = []
        labels = []
        # Use the globally defined class names for mapping
        label_to_index = {name: i for i, name in enumerate(self.class_names)}
        print(f"Looking for images in: {self.data_dir}")
        print(f"Class names expected: {self.class_names}")
        if not os.path.isdir(self.data_dir):
             raise ValueError(f"Data directory not found: {self.data_dir}. Please provide the correct path to the preprocessed images.")

        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Basic check for image file extensions
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(img_path)
                    labels.append(label_to_index[class_name])
                # else:
                #     print(f"Skipping non-image file: {img_path}")
        print(f"Found {len(image_paths)} image files belonging to {len(self.class_names)} classes.")
        if not image_paths:
             print(f"Warning: No image files found in {self.data_dir} for the specified classes.")
        return image_paths, labels

    def _prepare_datasets(self):
        image_paths, labels = self._get_paths_and_labels()
        if not image_paths:
            # Stop if no images were found
            raise ValueError(f"No images found in {self.data_dir} for classes {self.class_names}. Cannot create datasets.")

        # Split data into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=self.val_split,
            random_state=self.seed,
            stratify=labels # Ensure class distribution is similar in train/val
        )
        print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

        # Create tf.data datasets
        self.train_ds = self._create_dataset(train_paths, train_labels, is_training=True)
        self.val_ds = self._create_dataset(val_paths, val_labels, is_training=False)

    @tf.function
    def _load_and_preprocess(self, path, label):
        image = tf.io.read_file(path)
        try:
            # Try decoding as JPEG first
            image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
        except tf.errors.InvalidArgumentError:
            # If JPEG fails, try PNG
            try:
                image = tf.image.decode_png(image, channels=IMG_CHANNELS)
            except tf.errors.InvalidArgumentError:
                tf.print("Warning: Could not decode image:", path)
                image = tf.zeros([self.img_size[0], self.img_size[1], IMG_CHANNELS], dtype=tf.uint8)

        image = tf.image.resize(image, self.img_size, method='bilinear')
        # Preprocessing expects uint8 [0,255] and returns float32 [0,1]
        image = preprocess_image_tf(image)

        # Prepare labels for dual head
        # Head 1: Binary (0: normal, 1: diseased)
        is_normal = tf.cast(tf.math.equal(tf.gather(self.class_names, label), 'normal'), tf.float32)
        binary_label = 1.0 - is_normal # 0.0 if normal, 1.0 otherwise

        # Head 2: Multi-class (one-hot encoded) - Use self.num_classes
        multiclass_label = tf.one_hot(label, self.num_classes)

        return image, {'head1_output': binary_label, 'head2_output': multiclass_label}

    def _augment(self, image, labels):
        image = augment_image(image)
        return image, labels

    def _create_dataset(self, paths, labels, is_training=True):
        # Create a dataset of file paths and labels
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((path_ds, label_ds))

        # Load and preprocess images
        dataset = dataset.map(self._load_and_preprocess, num_parallel_calls=BUFFER_SIZE)

        # Cache, Shuffle, Augment, Batch, Prefetch
        dataset = dataset.cache() # Cache after loading/preprocessing
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(paths)) # Shuffle before batching
            # Apply augmentation only to the training set
            dataset = dataset.map(lambda img, lbl: (augment_image(img), lbl), num_parallel_calls=BUFFER_SIZE)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=BUFFER_SIZE) # Prefetch batches

        return dataset

    def get_datasets(self):
        return self.train_ds, self.val_ds

# --- Preprocessing Script ---
def preprocess_image(img_path):
    """Reads an image, resizes it, and converts to grayscale."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Read in color first
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            return None

        # Example preprocessing: Resize
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img_resized
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'Dataset')
    input_dir = os.path.join(dataset_dir, "train_images")
    output_dir_base = os.path.join(dataset_dir, "new_preprocessed_images") 
    metadata_file = os.path.join(dataset_dir, "meta_train.csv")

    # Create base output directory if it doesn't exist
    os.makedirs(output_dir_base, exist_ok=True)

    # Read metadata
    try:
        df = pd.read_csv(metadata_file)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}")
        return

    print(f"Found {len(df)} entries in metadata file.")

    # Process each image based on metadata
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preprocessing Images"):
        image_id = row['image_id']
        label = row['label']

        # Construct input and output paths
        input_img_path = os.path.join(input_dir, label, image_id) # Assumes images are in subfolders by label in train_images
        output_label_dir = os.path.join(output_dir_base, label)
        output_img_path = os.path.join(output_label_dir, image_id)

        # Create label-specific output directory if it doesn't exist
        os.makedirs(output_label_dir, exist_ok=True)

        # Check if input image exists before processing
        if not os.path.exists(input_img_path):
             # Fallback: Check if image is directly in input_dir (if not in label subfolder)
             input_img_path_alt = os.path.join(input_dir, image_id)
             if os.path.exists(input_img_path_alt):
                 input_img_path = input_img_path_alt
             else:
                 print(f"Warning: Input image not found at {input_img_path} or {input_img_path_alt}. Skipping.")
                 continue

        # Preprocess the image
        processed_img = preprocess_image(input_img_path)

        # Save the processed image if preprocessing was successful
        if processed_img is not None:
            try:
                cv2.imwrite(output_img_path, processed_img)
            except Exception as e:
                print(f"Error saving processed image {output_img_path}: {e}")

    print("\nData preprocessing complete.")
    print(f"Preprocessed images saved in: {output_dir_base}")

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIRECTORY = './Dataset/train_images' 
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    SEED = 42

    print("--- Data Preprocessing Script ---")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Class Names: {CLASS_NAMES}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Data Directory: {DATA_DIRECTORY}")

    # --- Find Duplicates (Optional) ---
    print("\n--- Checking for Duplicates ---")
    unique_files = find_and_remove_duplicates(DATA_DIRECTORY)
    print(f"Found {len(unique_files)} unique image files after check.")
    # Note: This function currently only finds duplicates, removal is commented out.
    # It also doesn't modify the dataset loading process below unless you integrate it.

    # --- Load Data ---
    print("\n--- Loading and Preparing Datasets ---")
    try:
        data_loader = DataLoader(
            data_dir=DATA_DIRECTORY,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
            seed=SEED
        )
        train_dataset, validation_dataset = data_loader.get_datasets()

        print("\n--- Dataset Details ---")
        print(f"Train dataset element spec: {train_dataset.element_spec}")
        print(f"Validation dataset element spec: {validation_dataset.element_spec}")

        # --- Inspect a Batch (Optional) ---
        print("\n--- Inspecting a Batch ---")
        for images, labels in train_dataset.take(1):
            print(f"Images batch shape: {images.shape}")
            print(f"Labels batch structure: {labels}") # Should show the two heads
            print(f"Binary labels batch shape: {labels['binary_output'].shape}")
            print(f"Multiclass labels batch shape: {labels['multiclass_output'].shape}")
            # Display the first image in the batch
            # plt.figure(figsize=(6, 6))
            # plt.imshow(images[0].numpy())
            # plt.title(f"Sample Image - Binary: {labels['binary_output'][0].numpy()}, MultiClass: {tf.argmax(labels['multiclass_output'][0]).numpy()}")
            # plt.axis("off")
            # plt.show()
            break # Only take one batch

        print("\nData loading and preprocessing setup complete.")

    except ValueError as e:
        print(f"\nError during data loading: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    # --- Run Preprocessing Script ---
    main()
