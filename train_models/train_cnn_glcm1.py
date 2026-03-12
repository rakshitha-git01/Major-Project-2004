import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# ================================
# CONFIG
# ================================
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, "datasets", "train")
test_dir = os.path.join(base_dir, "datasets", "test")
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 3

# ================================
# GLCM FEATURE EXTRACTION FUNCTION
# ================================
def extract_glcm_features(img_gray):
    glcm = graycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    ASM = graycoprops(glcm, 'ASM').flatten()
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
    return features

# ================================
# DATA AUGMENTATION
# ================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# ================================
# TRANSFER LEARNING CNN BRANCH (ResNet50)
# ================================
base_cnn = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling='avg'
)
base_cnn.trainable = False  # Freeze base model for initial training

cnn_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_cnn(cnn_input)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

glcm_input = Input(shape=(24,))
merged = Concatenate()([x, glcm_input])
x = Dense(64, activation='relu')(merged)
x = Dropout(0.3)(x)
final_output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[cnn_input, glcm_input], outputs=final_output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ================================
# DATA GENERATOR WITH GLCM
# ================================
def hybrid_generator(generator):
    for batch_x, batch_y in generator:
        glcm_features_batch = []
        for img in batch_x:
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            glcm_features = extract_glcm_features(gray)
            glcm_features_batch.append(glcm_features)
        glcm_features_batch = np.array(glcm_features_batch, dtype=np.float32)
        yield (batch_x.astype(np.float32), glcm_features_batch), batch_y.astype(np.float32)

output_signature = (
    (
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
)

train_hybrid_ds = tf.data.Dataset.from_generator(
    lambda: hybrid_generator(train_gen),
    output_signature=output_signature
)
val_hybrid_ds = tf.data.Dataset.from_generator(
    lambda: hybrid_generator(val_gen),
    output_signature=output_signature
)

# ================================
# TRAIN MODEL (Initial Training)
# ================================
steps_per_epoch = train_gen.samples // BATCH_SIZE
val_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_hybrid_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_hybrid_ds,
    validation_steps=val_steps,
    epochs=10
)

# ================================
# FINE-TUNE BASE MODEL
# ================================
base_cnn.trainable = True
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_ft = model.fit(
    train_hybrid_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_hybrid_ds,
    validation_steps=val_steps,
    epochs=10
)

# ================================
# PLOT RESULTS
# ================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'] + history_ft.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'] + history_ft.history['val_loss'], label="val_loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'] + history_ft.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'] + history_ft.history['val_accuracy'], label="val_acc")
plt.legend()
plt.title("Accuracy")

plt.show()