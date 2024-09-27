# Step 1: Setup Kaggle API
from google.colab import files
files.upload()  # Upload the kaggle.json file

# Install Kaggle API
!pip install -q kaggle

# Make a directory for Kaggle config
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Install TensorFlow and TensorFlow Hub
!pip install -q tensorflow tensorflow-hub

# Step 4: Download the correct dataset
!kaggle datasets download -d javaidahmadwani/lc25000

# Step 5: Unzip the downloaded dataset
!unzip lc25000.zip -d dataset

import os
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from google.colab import drive

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Define paths
data_dir = 'dataset/LC25000'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
checkpoint_dir = '/content/drive/My Drive/checkpoints/'  # Path to save checkpoints in Google Drive

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Step 3: Define image data generators with data augmentation for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 4: Load the GoogLeNet (InceptionV1) feature vector model from TensorFlow Hub
feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

# Step 5: Create a new model on top
model = tf.keras.Sequential([
    feature_extractor_layer,
    Dense(1024, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Step 6: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_path = os.path.join(checkpoint_dir, "googlenet-{epoch:02d}-{val_accuracy:.2f}.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Step 7: Check for existing checkpoints and load the latest one if available
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
else:
    print("No checkpoint found. Starting training from scratch.")

# Step 8: Train the model with the checkpoint callback
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10,
    callbacks=[checkpoint_callback]
)

# Save the final model
final_model_path = os.path.join(checkpoint_dir, 'googlenet.h5')
model.save('googlenet.h5')
from google.colab import files

# Download file to local machine
files.download('googlenet.h5')

# Step 9: Unfreeze the base model layers and fine-tune the model
feature_extractor_layer.trainable = True

# Step 10: Re-compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a new checkpoint callback for the fine-tuning phase
checkpoint_path_finetune = os.path.join(checkpoint_dir, "googlenet_finetune-{epoch:02d}-{val_accuracy:.2f}.h5")
checkpoint_callback_finetune = ModelCheckpoint(
    filepath=checkpoint_path_finetune,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Check for existing fine-tuning checkpoints and load the latest one if available
latest_checkpoint_finetune = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint_finetune:
    model.load_weights(latest_checkpoint_finetune)
    print(f"Resuming fine-tuning from checkpoint: {latest_checkpoint_finetune}")
else:
    print("No fine-tuning checkpoint found. Starting fine-tuning from scratch.")

# Step 11: Continue training (fine-tuning) with the new checkpoint callback
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10,
    callbacks=[checkpoint_callback_finetune]
)

# Save the fine-tuned model
final_model_finetune_path = os.path.join(checkpoint_dir, 'googlenet2.h5')
model.save('googlenet2.h5')
# Download file to local machine
files.download('googlenet2.h5')

# Step 12: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')