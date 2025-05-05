\
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import hashlib
import cv2  # Required for CLAHE and Gaussian Blur
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# from tflite_support.metadata_writers import image_classifier
# from tflite_support.metadata_writers import writer_utils
# Note: tflite_support needs to be installed separately: pip install tflite-support

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 8
BUFFER_SIZE = tf.data.AUTOTUNE # Dynamic buffer size

# Determine batch size based on GPU availability
BATCH_SIZE = BATCH_SIZE_GPU if len(tf.config.list_physical_devices('GPU')) > 0 else BATCH_SIZE_CPU

# Define paths (Adjust if necessary)
DATA_DIR = '/home/derrickle/Documents/RMIT/ML_Machine_Learning/Group_Project/Rice-Plant-Disease-Classification-/Dataset/preprocessed_images'
OUTPUT_DIR = '/home/derrickle/Documents/RMIT/ML_Machine_Learning/Group_Project/Rice-Plant-Disease-Classification-/output'
MODEL_NAME = 'rice_disease_classifier'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define class names based on folder structure
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(CLASS_NAMES)
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# --- 1. Data Preprocessing ---

# --- 1.1 Duplicate Removal ---
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
    print(f"Scan complete. Found {len(duplicates)} duplicates.")
    # Return list of unique files if needed, or just perform removal
    # For now, we just print duplicates. Actual removal is commented out.
    unique_files = list(hashes.values())
    print(f"Total unique files found: {len(unique_files)}")
    return unique_files # Or modify to return file paths grouped by class

# Run duplicate check (optional, uncomment to activate removal within the function)
# find_and_remove_duplicates(DATA_DIR)
# For DataLoader, we'll assume duplicates are handled or we proceed without removal for now.

# --- 1.2 Preprocessing Functions (using tf.py_function for OpenCV) ---
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
def preprocess_image(image):
    """Applies CLAHE and Gaussian Blur using tf.py_function."""
    # Decode image first if it's raw bytes
    # image = tf.image.decode_image(image, channels=3, expand_animations=False)
    # image.set_shape([None, None, 3]) # Set shape if decoded

    # Ensure image is float32 [0, 1] for TF ops, then back to uint8 for OpenCV
    if image.dtype != tf.uint8:
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8) # Convert back for CV

    # Apply CLAHE
    im_shape = image.shape
    [image,] = tf.py_function(apply_clahe_cv, [image], [tf.uint8])
    image.set_shape(im_shape) # py_function loses shape info

    # Apply Gaussian Blur
    [image,] = tf.py_function(apply_gaussian_blur_cv, [image], [tf.uint8])
    image.set_shape(im_shape)

    # Convert back to float32 for subsequent TF processing/model input
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

# --- 1.3 Augmentation Functions ---
def augment_image(image):
    """Applies random augmentations."""
    # Random Rotation (+/- 15 degrees)
    # Note: tf.contrib.image.rotate is deprecated. Using basic rotation or tfa.
    # Simple 90-degree rotations for now, replace with tfa.image.rotate if needed.
    # image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) # Random 0, 90, 180, 270
    # For +/- 15 degrees, tensorflow-addons is better:
    # import tensorflow_addons as tfa
    # angle = tf.random.uniform([], minval=-15, maxval=15) * (3.14159 / 180.0) # degrees to radians
    # image = tfa.image.rotate(image, angle, interpolation='BILINEAR')

    # Random Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image) # Optional, consider if relevant for rice leaves

    # HSV Jitter (+/- 20% saturation/value)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.2) # Adjust brightness (value)

    # Ensure values are clipped [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# --- 1.4 DataLoader Class ---
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
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    # Basic check for image file extensions
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(img_path)
                        labels.append(label_to_index[class_name])
        return image_paths, labels

    def _prepare_datasets(self):
        image_paths, labels = self._get_paths_and_labels()
        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir}. Check the path and structure.")

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
                # Handle other formats or raise an error if needed
                tf.print("Warning: Could not decode image:", path)
                # Return a placeholder or skip? For now, return zeros.
                image = tf.zeros([self.img_size[0], self.img_size[1], IMG_CHANNELS], dtype=tf.uint8)


        image = tf.image.resize(image, self.img_size, method='bilinear')
        image = tf.cast(image, tf.uint8) # Ensure uint8 for OpenCV functions

        # Apply CLAHE + Gaussian Blur
        image = preprocess_image(image) # Returns float32 [0,1]

        # Prepare labels for dual head
        # Head 1: Binary (0: normal, 1: diseased)
        # Assuming 'normal' is one of the class names
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
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=BUFFER_SIZE)

        if is_training:
            ds = ds.shuffle(buffer_size=len(paths)) # Shuffle before batching
            ds = ds.map(self._augment, num_parallel_calls=BUFFER_SIZE)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=BUFFER_SIZE) # Prefetch for performance
        return ds

    def get_datasets(self):
        return self.train_ds, self.val_ds

# --- 2. Model Architecture ---

# --- 2.1 Custom Layers/Blocks ---

# Efficient Channel Attention (ECA) Block
class EcaLayer(layers.Layer):
    """Efficient Channel Attention Layer"""
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        # Use Conv1D for adaptive kernel size based on channel dimension (as in ECA paper)
        # Or fixed kernel size Conv1D as simpler alternative:
        self.conv = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')

    def build(self, input_shape):
        # Input shape (batch, H, W, C)
        # After GAP: (batch, 1, 1, C)
        # Squeeze to (batch, 1, C) for Conv1D
        # Or transpose to (batch, C, 1) if using data_format='channels_last' in Conv1D
        super().build(input_shape)


    def call(self, inputs):
        x = self.gap(inputs) # (batch, 1, 1, C)
        # Prepare for Conv1D: (batch, C, 1)
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # -> (batch, C, 1, 1)
        x = tf.squeeze(x, axis=[2, 3]) # -> (batch, C)
        x = tf.expand_dims(x, axis=-1) # -> (batch, C, 1) needed for Conv1D

        x = self.conv(x) # (batch, C, 1)
        x = self.sigmoid(x) # (batch, C, 1)

        # Reshape back to (batch, 1, 1, C) for broadcasting
        x = tf.expand_dims(x, axis=1) # -> (batch, 1, C, 1)
        x = tf.transpose(x, perm=[0, 1, 3, 2]) # -> (batch, 1, 1, C)

        # Multiply with original input
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


# Inverted Residual Block (MobileNetV2 style)
def inverted_residual_block(inputs, filters, strides, expansion_factor, block_id):
    """MobileNetV2 Inverted Residual Block with Hardswish and optional ECA."""
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    x = inputs
    prefix = f'block_{block_id}_'

    # Expansion phase (Pointwise Convolution)
    if expansion_factor > 1:
        x = layers.Conv2D(
            filters=in_channels * expansion_factor,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand_conv'
        )(x)
        x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = layers.HardSwish(name=prefix + 'expand_hardswish')(x) # Use Hardswish

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding='same', # 'same' padding if strides=1, 'valid' if strides=2? Check MobileNetV2 paper. Usually 'same'.
        use_bias=False,
        name=prefix + 'depthwise_conv'
    )(x)
    x = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x)
    x = layers.HardSwish(name=prefix + 'depthwise_hardswish')(x) # Use Hardswish

    # Projection phase (Pointwise Convolution)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project_conv'
    )(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)

    # Add ECA layer after the bottleneck projection
    x = EcaLayer(name=prefix + 'eca')(x)

    # Residual connection only if strides=1 and input/output channels match
    if strides == 1 and in_channels == filters:
        x = layers.Add(name=prefix + 'add')([inputs, x])

    # NOTE: RepMLP layers would be inserted here or after a sequence of blocks
    # Example: if block_id % 2 == 0: x = RepMLPLayer(...)(x)
    # Skipping RepMLP due to complexity for this example.

    return x

# --- 2.2 Model Definition ---
def MobileNetDualHead(input_shape, num_classes, include_top=True, name='MobileNetDualHead'):
    """Builds the custom MobileNet-style model with dual heads."""
    inputs = keras.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='initial_conv')(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.HardSwish(name='initial_hardswish')(x) # Use Hardswish

    # Sequence of Inverted Residual Blocks
    # Parameters follow MobileNetV2 structure generally (filters, strides, expansion_factor)
    # Block 0
    x = inverted_residual_block(x, filters=16, strides=1, expansion_factor=1, block_id=0)
    # Block 1, 2
    x = inverted_residual_block(x, filters=24, strides=2, expansion_factor=6, block_id=1)
    x = inverted_residual_block(x, filters=24, strides=1, expansion_factor=6, block_id=2)
    # Block 3, 4, 5
    x = inverted_residual_block(x, filters=32, strides=2, expansion_factor=6, block_id=3)
    x = inverted_residual_block(x, filters=32, strides=1, expansion_factor=6, block_id=4)
    x = inverted_residual_block(x, filters=32, strides=1, expansion_factor=6, block_id=5)
    # Block 6, 7, 8, 9
    x = inverted_residual_block(x, filters=64, strides=2, expansion_factor=6, block_id=6)
    x = inverted_residual_block(x, filters=64, strides=1, expansion_factor=6, block_id=7)
    x = inverted_residual_block(x, filters=64, strides=1, expansion_factor=6, block_id=8)
    x = inverted_residual_block(x, filters=64, strides=1, expansion_factor=6, block_id=9)
    # Block 10, 11, 12
    x = inverted_residual_block(x, filters=96, strides=1, expansion_factor=6, block_id=10)
    x = inverted_residual_block(x, filters=96, strides=1, expansion_factor=6, block_id=11)
    x = inverted_residual_block(x, filters=96, strides=1, expansion_factor=6, block_id=12)
    # Block 13, 14, 15
    x = inverted_residual_block(x, filters=160, strides=2, expansion_factor=6, block_id=13)
    x = inverted_residual_block(x, filters=160, strides=1, expansion_factor=6, block_id=14)
    x = inverted_residual_block(x, filters=160, strides=1, expansion_factor=6, block_id=15)
    # Block 16
    x = inverted_residual_block(x, filters=320, strides=1, expansion_factor=6, block_id=16)

    # Final layers before heads
    x = layers.Conv2D(1280, kernel_size=1, padding='same', use_bias=False, name='final_conv')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    x = layers.HardSwish(name='final_hardswish')(x)
    features = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    if include_top:
        # Head 1: Binary Classification (Healthy/Diseased)
        head1 = layers.Dense(1, activation='sigmoid', name='head1_output')(features)

        # Head 2: Multi-class Disease Identification
        head2 = layers.Dense(num_classes, activation='softmax', name='head2_output')(features)

        model = keras.Model(inputs=inputs, outputs={'head1_output': head1, 'head2_output': head2}, name=name)
    else:
        # Return features if include_top is False
        model = keras.Model(inputs=inputs, outputs=features, name=name + '_base')

    return model

# --- 3. Training Protocol ---

# --- 3.1 Loss Function (Focal Loss) ---
# Using tf.keras.losses implementation if available, otherwise define manually
# Note: TF Addons has SigmoidFocalCrossEntropy. For Softmax output, need categorical version.
# Let's define a categorical focal loss manually.
def categorical_focal_loss(alpha=0.8, gamma=2.0):
    """
    Categorical Focal Loss implementation.
    Usage: compile(loss={'head2_output': categorical_focal_loss(alpha=0.8, gamma=2.0)})
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: one-hot encoded labels
        # y_pred: softmax probabilities
        epsilon = tf.keras.backend.epsilon() # 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) # Avoid log(0)

        # Calculate cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal loss component
        loss = alpha * tf.pow(1. - y_pred, gamma) * cross_entropy

        # Sum over classes, mean over batch
        return tf.reduce_sum(loss, axis=-1)

    return loss_fn

# --- 3.2 Optimizer ---
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# --- 3.3 Callbacks ---
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_head2_output_accuracy', # Monitor validation accuracy of the main task
    patience=15, # Increased patience slightly
    verbose=1,
    restore_best_weights=True # Restore model weights from the epoch with the best value
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_head2_output_accuracy',
    factor=0.2, # Reduce LR by a factor of 5
    patience=5,  # Reduce LR if no improvement after 5 epochs
    verbose=1,
    min_lr=1e-6 # Minimum learning rate
)

# Checkpoint saving
checkpoint_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best.keras") # Use .keras format
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_head2_output_accuracy',
    save_best_only=True,
    save_weights_only=False, # Save entire model
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# --- 4. Training Loop ---
print("\n--- Initializing Data Loaders ---")
data_loader = DataLoader(DATA_DIR, IMG_SIZE, BATCH_SIZE, CLASS_NAMES, val_split=0.2)
train_ds, val_ds = data_loader.get_datasets()

# Verify dataset output structure
print("\n--- Verifying Dataset Output ---")
for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Labels batch type:", type(labels))
    print("Label keys:", labels.keys())
    print("Head 1 label batch shape:", labels['head1_output'].shape)
    print("Head 2 label batch shape:", labels['head2_output'].shape)
    # Display one image from the batch
    plt.figure(figsize=(5, 5))
    plt.imshow(images[0].numpy())
    plt.title(f"Head1: {labels['head1_output'][0].numpy()}, Head2: {tf.argmax(labels['head2_output'][0]).numpy()}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_batch_image.png"))
    print(f"Saved sample batch image to {OUTPUT_DIR}")
    # plt.show() # Uncomment to display inline if running interactively


print("\n--- Building Model ---")
model = MobileNetDualHead(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES)
model.summary()

print("\n--- Compiling Model ---")
model.compile(
    optimizer=optimizer,
    loss={
        'head1_output': 'binary_crossentropy', # Standard BCE for binary head
        'head2_output': categorical_focal_loss(alpha=0.8, gamma=2.0) # Focal loss for multi-class head
    },
    loss_weights={ # Optional: Weight losses if one head is more important
        'head1_output': 0.5,
        'head2_output': 1.0
    },
    metrics={
        'head1_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        'head2_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    }
)

print("\n--- Starting Training ---")
EPOCHS = 50 # Adjust as needed
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n--- Training Finished ---")

# --- 5. Evaluation ---
print("\n--- Evaluating Model on Validation Set ---")
# Load best model saved by checkpoint
print(f"Loading best model from: {checkpoint_path}")
best_model = tf.keras.models.load_model(checkpoint_path, custom_objects={
    'EcaLayer': EcaLayer,
    'categorical_focal_loss': categorical_focal_loss(alpha=0.8, gamma=2.0), # Need to pass the loss func object
    'HardSwish': layers.HardSwish # Or tf.nn.hard_swish if used directly
})

# Evaluate the best model
results = best_model.evaluate(val_ds, verbose=1)
print("\nValidation Set Evaluation Results:")
# Create a dictionary of metrics for easier reading
metric_names = best_model.metrics_names
results_dict = dict(zip(metric_names, results))
print(results_dict)

# Generate predictions for confusion matrix (on validation set)
print("\n--- Generating Predictions for Confusion Matrix ---")
val_labels_list = []
val_predictions_list_head1 = []
val_predictions_list_head2 = []

# It's more efficient to predict on the dataset directly
predictions = best_model.predict(val_ds)
pred_head1 = predictions['head1_output']
pred_head2 = predictions['head2_output'] # These are probabilities

# Need true labels - iterate through the validation dataset
print("Extracting true labels from validation set...")
for _, labels_batch in val_ds:
    val_labels_list.append(labels_batch['head2_output'].numpy()) # Get head2 labels (one-hot)

# Concatenate all batches
true_labels_one_hot = np.concatenate(val_labels_list, axis=0)
true_labels_indices = np.argmax(true_labels_one_hot, axis=1) # Convert one-hot to indices

# Get predicted class indices for head 2
predicted_labels_indices_head2 = np.argmax(pred_head2, axis=1)

# Calculate and plot confusion matrix for Head 2 (Disease Classification)
print("\n--- Confusion Matrix (Head 2: Disease Classification) ---")
cm = confusion_matrix(true_labels_indices, predicted_labels_indices_head2)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Head 2 (Disease Classification)')
cm_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix_head2.png")
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}")
# plt.show()

# Print classification report for Head 2
print("\nClassification Report (Head 2):")
print(classification_report(true_labels_indices, predicted_labels_indices_head2, target_names=CLASS_NAMES))

# Optional: Evaluate Head 1 (Binary Classification)
print("\n--- Evaluation (Head 1: Binary Classification) ---")
true_labels_binary = (true_labels_indices != CLASS_NAMES.index('normal')).astype(int) # 0 if normal, 1 otherwise
predicted_labels_binary = (pred_head1 > 0.5).astype(int).flatten() # Threshold sigmoid output

print("\nConfusion Matrix (Head 1: Healthy/Diseased):")
cm_head1 = confusion_matrix(true_labels_binary, predicted_labels_binary)
print(cm_head1)
print("\nClassification Report (Head 1):")
print(classification_report(true_labels_binary, predicted_labels_binary, target_names=['Healthy (normal)', 'Diseased']))


# Plot training history
def plot_history(history, output_dir, model_name):
    """Plots training and validation loss and accuracy."""
    # Plot Head 1 Accuracy
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['head1_output_accuracy'], label='Train Accuracy (Head 1)')
    plt.plot(history.history['val_head1_output_accuracy'], label='Val Accuracy (Head 1)')
    plt.title('Head 1: Binary Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Head 1 Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['head1_output_loss'], label='Train Loss (Head 1)')
    plt.plot(history.history['val_head1_output_loss'], label='Val Loss (Head 1)')
    plt.title('Head 1: Binary Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Head 2 Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history.history['head2_output_accuracy'], label='Train Accuracy (Head 2)')
    plt.plot(history.history['val_head2_output_accuracy'], label='Val Accuracy (Head 2)')
    plt.title('Head 2: Disease Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Head 2 Loss
    plt.subplot(2, 2, 4)
    plt.plot(history.history['head2_output_loss'], label='Train Loss (Head 2)')
    plt.plot(history.history['val_head2_output_loss'], label='Val Loss (Head 2)')
    plt.title('Head 2: Disease Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    # plt.show()

plot_history(history, OUTPUT_DIR, MODEL_NAME)


# --- 6. Deployment Readiness (TFLite Conversion) ---
print("\n--- Converting Model to TFLite (FP16 Quantization) ---")
try:
    # Convert the best saved Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

    # Enable FP16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.lite.float16]

    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_fp16.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model (FP16) saved to: {tflite_model_path}")
    print(f"TFLite model size: {os.path.getsize(tflite_model_path) / (1024):.2f} KB")

    # --- Add Metadata (Optional but recommended) ---
    # Requires pip install tflite-support
    # try:
    #     print("\n--- Adding Metadata to TFLite Model ---")
    #     # Create metadata writer
    #     writer = image_classifier.MetadataWriter.create_for_inference(
    #         writer_utils.load_file(tflite_model_path),
    #         input_norm_mean=[0.0], # Assuming input is [0,1] float
    #         input_norm_std=[1.0],
    #         label_file_paths=[os.path.join(OUTPUT_DIR, "labels.txt")] # Need to create this file
    #     )
    #
    #     # Create labels file
    #     labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    #     with open(labels_path, "w") as f:
    #         f.write("\n".join(CLASS_NAMES))
    #     print(f"Labels file created at: {labels_path}")
    #
    #     # Populate metadata (check TFLite Support docs for dual output handling)
    #     # Metadata for multi-output models might require more specific handling.
    #     # This example assumes metadata primarily for the classification head (head 2).
    #     # You might need to adjust based on how you intend to use the model.
    #
    #     # writer.associate_file(...) # Add associated files like labels
    #
    #     # Populate output tensor metadata (assuming head2 is index 1 if model outputs are ordered)
    #     # output_meta = writer.get_output_tensor_metadata()[1] # Adjust index if needed
    #     # output_meta.name = "disease_probabilities"
    #     # output_meta.description = "Probabilities for each disease class"
    #     # writer.populate() # This might fail if structure isn't standard single output classification
    #
    #     # Save metadata to the model file
    #     # populated_model_buffer = writer.populate()
    #     # tflite_metadata_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_fp16_metadata.tflite")
    #     # writer_utils.save_file(populated_model_buffer, tflite_metadata_model_path)
    #     # print(f"TFLite model with metadata saved to: {tflite_metadata_model_path}")
    #
    # except ImportError:
    #     print("\nSkipping TFLite metadata: tflite-support library not found.")
    #     print("Install it using: pip install tflite-support")
    # except Exception as e:
    #     print(f"\nError adding TFLite metadata: {e}")
    #     print("Metadata for multi-output models might require specific handling.")

except Exception as e:
    print(f"\nError during TFLite conversion: {e}")

print("\n--- Script Finished ---")

