# train_cnn_glcm_full.py
# Run: python train_cnn_glcm_full.py
# Make sure virtualenv active and dependencies installed.

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import exposure

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input

# --------------------- CONFIG ---------------------
BASE_DIR = Path(__file__).resolve().parent  # backend folder
TRAIN_DIR = BASE_DIR / "datasets" / "train"      # <-- MODIFIED
TEST_DIR = BASE_DIR / "datasets" / "test"        # <-- MODIFIED
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)   # input size for CNN feature extractor
ROI_SIZE = (128, 128)   # size for GLCM and CNN ROI if using direct CNN on ROIs
BATCH_PRINT = 50        # progress print frequency

LABELS = ["Normal", "Benign", "Malignant"]  # folder names and mapping order
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}

# --------------------- UTIL: Preprocess ---------------------
def preprocess_image_gray(img_path):
    """Read grayscale, denoise and apply CLAHE. Returns uint8 image (0-255)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {img_path}")
    # Denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

# --------------------- UTIL: Lung segmentation ---------------------
def segment_lung_mask(img):
    """
    Return a binary mask of the lung region.
    Uses thresholding + morphological ops to get largest components (lungs).
    """
    # Otsu threshold (image should be uint8)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert if background is white
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    # Remove small noise, close holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Keep largest two components (lungs)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img, dtype=np.uint8) * 255
    # Sort by area descending and fill top 2
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(img, dtype=np.uint8)
    for c in contours[:2]:
        cv2.drawContours(mask, [c], -1, color=255, thickness=-1)
    # Morphological opening to smooth
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# --------------------- UTIL: Candidate ROI extraction ---------------------
def extract_candidate_rois(img_gray, mask, min_area=50, max_area=5000):
    """
    Find candidate nodules inside lung mask:
    - Use adaptive threshold inside lung mask, find contours, filter by area and circularity.
    - Returns list of cropped ROI images (uint8) resized to ROI_SIZE.
    """
    # Masked lung region
    lung = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    # Equalize and adaptive threshold to highlight bright spots (nodules)
    blurred = cv2.GaussianBlur(lung, (3,3), 0)
    # Use local adaptive thresholding
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    # Mask again
    th = cv2.bitwise_and(th, th, mask=mask)
    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # Find contours (blobs)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        # Compute circularity to prefer nodule-like shapes
        perimeter = cv2.arcLength(cnt, True)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Accept somewhat circular blobs; threshold can be tuned
        if circularity < 0.1:
            continue
        # Expand bounding box a bit
        pad = int(0.2 * max(w,h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(img_gray.shape[1], x + w + pad); y1 = min(img_gray.shape[0], y + h + pad)
        roi = img_gray[y0:y1, x0:x1]
        # Resize to ROI_SIZE
        roi = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)
        rois.append(roi)
    return rois

# --------------------- UTIL: fallback ROI (whole lung) ---------------------
def lung_roi_fallback(img_gray, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = img_gray[y:y+h, x:x+w]
    if roi.size == 0:
        roi = cv2.resize(img_gray, ROI_SIZE)
    else:
        roi = cv2.resize(roi, ROI_SIZE)
    return roi

# --------------------- GLCM features ---------------------
def extract_glcm_features(roi):
    """
    roi: grayscale uint8 (0-255), expected shape ROI_SIZE
    returns flattened features (here 4 props x 4 angles = 16)
    """
    # Reduce gray levels for stable GLCM
    img_q = (roi // 16).astype(np.uint8)  # 16 levels (0-15)
    glcm = graycomatrix(img_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=16, symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    feats = []
    for p in props:
        vals = graycoprops(glcm, p).flatten()  # length 4 (angles)
        feats.extend(vals.tolist())
    return np.array(feats, dtype=np.float32)  # length 16

# --------------------- CNN feature extractor ---------------------
def build_cnn_feature_extractor(img_size=IMG_SIZE):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3), pooling='avg')
    base.trainable = False
    inp = Input(shape=(img_size[0], img_size[1], 3))
    out = base(inp)
    feat_model = Model(inputs=inp, outputs=out)
    return feat_model

def extract_cnn_feature_from_roi(roi_gray, feat_model):
    """
    Convert gray ROI to 3-channel, resize to IMG_SIZE, run feature extractor, return 1D vector.
    roi_gray: ROI_SIZE (128,128)
    """
    # Resize to IMG_SIZE for CNN
    img_small = cv2.resize(roi_gray, IMG_SIZE)
    # Convert to 3-channel
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)
    # Preprocess for MobileNetV2
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb.astype(np.float32))
    arr = np.expand_dims(arr, axis=0)
    feat = feat_model.predict(arr, verbose=0)
    return feat.flatten()

# --------------------- Dataset loader (per-image -> single-feature vector) ---------------------
def process_image_to_feature_vector(img_path, feat_model):
    """
    For a single image:
    - preprocess
    - segment lungs
    - extract candidate ROIs
    - if multiple ROIs, choose best (largest area) or average features (we'll average for robustness)
    - build fused feature vector [cnn_feat(avg), glcm_feat(avg)]
    """
    img_gray = preprocess_image_gray(img_path)
    mask = segment_lung_mask(img_gray)
    rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)

    roi_feats_cnn = []
    roi_feats_glcm = []
    if not rois:
        # fallback to whole lung ROI
        roi = lung_roi_fallback(img_gray, mask)
        rois = [roi]

    for roi in rois:
        try:
            glcm_f = extract_glcm_features(roi)           # 16-d
            cnn_f = extract_cnn_feature_from_roi(roi, feat_model)  # ~1280-d depending on MobileNetV2 pooling
            roi_feats_glcm.append(glcm_f)
            roi_feats_cnn.append(cnn_f)
        except Exception as e:
            # skip problematic ROI
            continue

    if not roi_feats_cnn:  # safety fallback
        roi = lung_roi_fallback(img_gray, mask)
        roi_feats_glcm = [extract_glcm_features(roi)]
        roi_feats_cnn = [extract_cnn_feature_from_roi(roi, feat_model)]

    # Aggregate (mean) across ROIs
    cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
    glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)

    # Optionally normalize glcm_vec scale here
    fused = np.hstack([cnn_vec, glcm_vec])
    return fused

# --------------------- Build dataset (train/test) ---------------------
def build_feature_dataset(folder, feat_model, max_samples_per_class=None):
    X = []
    y = []
    folder = Path(folder)
    for label in LABELS:
        cls_dir = folder / label
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping")
            continue
        files = list(cls_dir.glob("*"))
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        for i, f in enumerate(tqdm(files, desc=f"Processing {label}", leave=False)):
            try:
                vec = process_image_to_feature_vector(f, feat_model)
                X.append(vec)
                y.append(LABEL_MAP[label])
            except Exception as e:
                print(f"Skipped {f} due to {e}")
                continue
    if not X:
        raise ValueError("No images found in any class folders. Check your dataset structure.")
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

# --------------------- MAIN TRAIN/EVAL ---------------------
def main():
    print("Building CNN feature extractor...")
    feat_model = build_cnn_feature_extractor(IMG_SIZE)
    # save feature extractor for reuse
    feat_model.save(str(MODEL_DIR / "cnn_feature_extractor.h5"))
    print("Saved CNN feature extractor to", MODEL_DIR / "cnn_feature_extractor.h5")

    print("\n--- Building TRAIN dataset features ---")
    X_train, y_train = build_feature_dataset(TRAIN_DIR, feat_model)
    print("Train features shape:", X_train.shape, "Train labels:", y_train.shape)

    print("\n--- Building TEST dataset features ---")
    X_test, y_test = build_feature_dataset(TEST_DIR, feat_model)
    print("Test features shape:", X_test.shape, "Test labels:", y_test.shape)

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.gz")
    print("Saved scaler to", MODEL_DIR / "scaler.gz")

    # Train SVM
    print("\nTraining SVM (may take a while depending on feature size)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_s, y_train)
    joblib.dump(svm, MODEL_DIR / "svm_fused.pkl")
    print("Saved SVM to", MODEL_DIR / "svm_fused.pkl")

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy: {:.3f}".format(acc))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=LABELS))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(LABELS)), LABELS)
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
if __name__ == "__main__":
    main()