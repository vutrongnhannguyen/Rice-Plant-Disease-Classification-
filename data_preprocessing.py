import tensorflow as tf
import numpy as np
import os
import hashlib
import cv2
from sklearn.model_selection import train_test_split

# --- Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BUFFER_SIZE = tf.data.AUTOTUNE # Dynamic buffer size

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

# --- Preprocessing Functions (using tf.py_function for OpenCV) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe_cv(image):
    """Applies CLAHE using OpenCV."""
    # Assumes image is uint8 BGR
    if image.shape[-1] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else: # Grayscale
        final_img = clahe.apply(image)
    return final_img

def apply_gaussian_blur_cv(image):
    """Applies Gaussian Blur using OpenCV."""
    return cv2.GaussianBlur(image, (5, 5), 0)

@tf.function
def preprocess_image_tf(image):
    """Applies CLAHE and Gaussian Blur using tf.py_function."""
    # Ensure image is uint8 [0, 255] for OpenCV functions
    if image.dtype != tf.uint8:
         # If input is float [0,1], scale to [0,255] and cast
        if tf.reduce_max(image) <= 1.0:
            image = tf.cast(image * 255.0, tf.uint8)
        else:
            image = tf.cast(image, tf.uint8)

    im_shape = image.shape
    # Apply CLAHE
    [image,] = tf.py_function(apply_clahe_cv, [image], [tf.uint8])
    image.set_shape(im_shape) # py_function loses shape info

    # Apply Gaussian Blur
    [image,] = tf.py_function(apply_gaussian_blur_cv, [image], [tf.uint8])
    image.set_shape(im_shape)

    # Convert back to float32 [0, 1] for model input
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
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
        # tf.print("Warning: tensorflow-addons not installed. Skipping rotation augmentation.")
        pass # Silently skip rotation if tfa not available

    # Ensure values are clipped [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# --- DataLoader Class ---
class DataLoader:
    def __init__(self, data_dir, img_size, batch_size, class_names, val_split=0.2, seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_split = val_split
        self.seed = seed
        self._prepare_datasets()

    def _get_paths_and_labels(self):
        image_paths = []
        labels = []
        label_to_index = {name: i for i, name in enumerate(self.class_names)}
        print(f"Looking for images in: {self.data_dir}")
        print(f"Class names expected: {self.class_names}")
        if not os.path.isdir(self.data_dir):
             raise ValueError(f"Data directory not found: {self.data_dir}")

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
        print(f"Found {len(image_paths)} image files.")
        return image_paths, labels

    def _prepare_datasets(self):
        image_paths, labels = self._get_paths_and_labels()
        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir} for classes {self.class_names}. Check the path and structure.")

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

        # Head 2: Multi-class (one-hot encoded)
        multiclass_label = tf.one_hot(label, self.num_classes)

        return image, {'head1_output': binary_label, 'head2_output': multiclass_label}

    def _augment(self, image, labels):
        image = augment_image(image)
        return image, labels

    def _create_dataset(self, paths, labels, is_training):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        # Use experimental_deterministic=False for potential speedup if order doesn't matter after shuffle
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=BUFFER_SIZE)

        if is_training:
            # Consider shuffling before mapping if _load_and_preprocess is slow
            ds = ds.shuffle(buffer_size=max(1000, len(paths))) # Use a reasonably large buffer
            ds = ds.map(self._augment, num_parallel_calls=BUFFER_SIZE)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=BUFFER_SIZE) # Prefetch for performance
        return ds

    def get_datasets(self):
        return self.train_ds, self.val_ds
