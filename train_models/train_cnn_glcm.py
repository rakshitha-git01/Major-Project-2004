# train_cnn_glcm.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# ================================
# CONFIG
# ================================
base_dir = os.path.dirname(os.path.abspath(__file__))   # backend folder path
train_dir = os.path.join(base_dir, "datasets", "train")
test_dir = os.path.join(base_dir, "datasets", "test")
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 5   # increase later (start small to test)
NUM_CLASSES = 3   # benign, malignant, normal

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
# DATA PREPROCESSING
# ================================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ================================
# CNN BRANCH (Functional API)
# ================================
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    return inputs, x

cnn_input, cnn_output = build_cnn((IMG_HEIGHT, IMG_WIDTH, 3))
glcm_input = Input(shape=(24,))   # 6 features × 4 directions = 24
merged = Concatenate()([cnn_output, glcm_input])
x = Dense(64, activation='relu')(merged)
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
# TRAIN MODEL
# ================================
steps_per_epoch = train_gen.samples // BATCH_SIZE
val_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_hybrid_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_hybrid_ds,
    validation_steps=val_steps,
    epochs=EPOCHS
)

# ================================
# PLOT RESULTS
# ================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.legend()
plt.title("Accuracy")

plt.show()