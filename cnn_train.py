import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ====== CONFIG ======
IMAGE_SIZE = (220, 220)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2

# ====== CORRECTED PATHS ======
elephant_dir = "Elephant"
others_dir = "others"

# ====== PREPROCESSING FUNCTIONS ======
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.adjust_contrast(image, 2)
    image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def prepare_dataset(file_paths, labels, augment_data=False):
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(200)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ====== LOAD IMAGE FILE PATHS ======
elephant_files = [os.path.join(elephant_dir, f) for f in os.listdir(elephant_dir) if f.lower().endswith(('jpg','jpeg','png'))]
others_files = [os.path.join(others_dir, f) for f in os.listdir(others_dir) if f.lower().endswith(('jpg','jpeg','png'))]

all_files = elephant_files + others_files
all_labels = [1]*len(elephant_files) + [0]*len(others_files)

train_files, val_files, train_labels, val_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42)

train_ds = prepare_dataset(train_files, train_labels, augment_data=True)
val_ds = prepare_dataset(val_files, val_labels)

# ====== CNN MODEL ======
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(220, 220, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ====== CALLBACKS ======
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'best_elephant_classifier_103elephant.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

class TqdmProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.progbar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

    def on_batch_end(self, batch, logs=None):
        self.progbar.update(1)
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        self.progbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        self.progbar.close()
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f} sec - "
              f"loss: {logs.get('loss'):.4f} - val_loss: {logs.get('val_loss'):.4f} - "
              f"acc: {logs.get('accuracy'):.4f} - val_acc: {logs.get('val_accuracy'):.4f}")

tqdm_bar = TqdmProgressBar()

# ====== TRAIN ======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop, checkpoint_cb, tqdm_bar]
)

# ====== SAVE FINAL MODEL ======
model.save('final_elephant_model.h5')
print("✅ Final model saved as final_elephant_model.h5")
# ====== EVALUATE MODEL ======
loss, accuracy = model.evaluate(val_ds)
print(f"📊 Validation loss: {loss:.4f} - Validation accuracy: {accuracy:.4f}")  