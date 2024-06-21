#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths and constants
healthy_path = "C:/Users/Lenovo/Coral/healthy_corals"
bleached_path = "C:/Users/Lenovo/Coral/bleached_corals"
categories = ['healthy_corals', 'bleached_corals']
IMG_SIZE = (64, 64)

# Function to load and process images
def load_images():
    unique_images = []
    unique_labels = []
    image_paths = {}
    duplicates = []

    for category in categories:
        path = os.path.join("C:/Users/Lenovo/Coral", category)
        class_num = categories.index(category)

        for img_name in tqdm(os.listdir(path), desc=f"Loading {category} images"):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).resize(IMG_SIZE)
            img_array = np.array(img)

            is_duplicate = False
            for unique_img in unique_images:
                if np.array_equal(img_array, unique_img):
                    duplicates.append((img_path, image_paths[tuple(unique_img.flatten())]))
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_images.append(img_array)
                unique_labels.append(class_num)
                image_paths[tuple(img_array.flatten())] = img_path

    images = np.array(unique_images)
    labels = np.array(unique_labels)

    return images, labels, duplicates

# Function to augment images
def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_images.append(np.array(ImageOps.mirror(Image.fromarray(img))))
        augmented_images.append(np.array(Image.fromarray(img).rotate(90)))
        augmented_labels.extend([label, label, label])

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

# Function to convert images to grayscale
def convert_to_grayscale(images):
    grayscale_images = np.array([np.array(Image.fromarray(img).convert('L')) for img in images])
    return grayscale_images

# Function to adjust contrast and brightness
def adjust_images(images):
    adjusted_images = []
    for img in images:
        img_pil = Image.fromarray(img)
        contrast_enhancer = ImageEnhance.Contrast(img_pil)
        img_contrast = contrast_enhancer.enhance(2)
        brightness_enhancer = ImageEnhance.Brightness(img_contrast)
        img_bright = brightness_enhancer.enhance(1.5)
        adjusted_images.append(np.array(img_bright))
    return np.array(adjusted_images)

# Function to flatten images
def flatten_images(images):
    flattened_images = [np.array(img).flatten() for img in images]
    return np.array(flattened_images)

# Function to create DataFrame from images
def create_dataframe(images, labels):
    dataset = {'Label': labels}
    for i in range(images.shape[1]):
        dataset[f'Pixel_{i}'] = images[:, i]
    return pd.DataFrame(dataset)

# Function to train and evaluate model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return best_model, metrics, grid_search.best_params_

# Load and process images
images, labels, duplicates = load_images()
augmented_images, augmented_labels = augment_images(images, labels)
grayscale_images = convert_to_grayscale(augmented_images)
adjusted_images = adjust_images(grayscale_images)
flattened_images = flatten_images(adjusted_images)
df = create_dataframe(flattened_images, augmented_labels)

# Split data
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title('Coral Image Classification')

st.write("## Duplicate Images")
if len(duplicates) > 0:
    for dup in duplicates:
        img1_path, img2_path = dup
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=f'Duplicate: {img1_path}')
        with col2:
            st.image(img2, caption=f'Original: {img2_path}')
else:
    st.write("No duplicate images found")

st.write("## Data Summary")
st.write(f"Loaded {len(images)} unique images")
st.write(f"Found {len(duplicates)} duplicate images")
st.write(f"Remaining images after duplicate removal: {len(flattened_images)}")

# Train and evaluate models
st.write("## Model Training and Evaluation")

models = {
    "Logistic Regression": (LogisticRegression(), {'model__C': [0.1, 1, 10], 'model__solver': ['liblinear']}),
    "Random Forest": (RandomForestClassifier(), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5]}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}),
    "Decision Tree": (DecisionTreeClassifier(), {'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5, 10]}),
    "Naive Bayes": (GaussianNB(), {})
}

for model_name, (model, param_grid) in models.items():
    with st.spinner(f'Training {model_name}...'):
        best_model, metrics, best_params = train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid)
        st.write(f"### {model_name}")
        st.write(f"Best Parameters: {best_params}")
        st.write(f"Accuracy: {metrics['accuracy']}")
        st.write(f"Precision: {metrics['precision']}")
        st.write(f"Recall: {metrics['recall']}")
        st.write(f"F1 Score: {metrics['f1_score']}")
        st.write(f"Confusion Matrix: \n{metrics['confusion_matrix']}")

st.write("## Visualizations")

# Plotting images
st.write("### Original, Augmented, Grayscale, and Adjusted Images")
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs[0, 0].imshow(images[0])
axs[0, 0].set_title("Original Image")
axs[0, 1].imshow(augmented_images[1])
axs[0, 1].set_title("Augmented Image")
axs[0, 2].imshow(grayscale_images[2], cmap='gray')
axs[0, 2].set_title("Grayscale Image")
axs[0, 3].imshow(adjusted_images[3], cmap='gray')
axs[0, 3].set_title("Adjusted Image")
axs[1, 0].imshow(images[4])
axs[1, 0].set_title("Original Image")
axs[1, 1].imshow(augmented_images[5])
axs[1, 1].set_title("Augmented Image")
axs[1, 2].imshow(grayscale_images[6], cmap='gray')
axs[1, 2].set_title("Grayscale Image")
axs[1, 3].imshow(adjusted_images[7], cmap='gray')
axs[1, 3].set_title("Adjusted Image")
st.pyplot(fig)

# Label distribution
st.write("### Label Distribution")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x=labels, ax=ax[0])
ax[0].set_title('Original Dataset')
sns.countplot(x=augmented_labels, ax=ax[1])
ax[1].set_title('Augmented Dataset')
st.pyplot(fig)

# Display sample images with labels
st.write("### Sample Images with Labels")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i in np.arange(0, 10):
    axes[i].imshow(images[i])
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')
st.pyplot(fig)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i in np.arange(0, 10):
    axes[i].imshow(grayscale_images[i], cmap='gray')
    axes[i].set_title(f'Label: {augmented_labels[i]}')
    axes[i].axis('off')
st.pyplot(fig)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i in np.arange(0, 10):
    axes[i].imshow(adjusted_images[i], cmap='gray')
    axes[i].set_title(f'Label: {augmented_labels[i]}')
    axes[i].axis('off')
st.pyplot(fig)

