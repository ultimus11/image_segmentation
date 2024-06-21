import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the image
image_path = 'image.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
def thresholding(image):
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresholded

# Edge-Based Segmentation (Canny)
def edge_based_segmentation(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Region-Based Segmentation (Watershed)
def watershed_segmentation(image, gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 0] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image

# Clustering-Based Segmentation (K-means)
def kmeans_segmentation(image):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

# Deep Learning-Based Segmentation (U-Net)
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv2))
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    model = models.Model(inputs, conv10)
    return model

def display_image(title, image, is_gray=False):
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Thresholding
    thresholded = thresholding(gray)
    display_image('Thresholded Image', thresholded, is_gray=True)

    # Edge-Based Segmentation
    edges = edge_based_segmentation(gray)
    display_image('Edge Detected Image', edges, is_gray=True)

    # Region-Based Segmentation
    watershed_result = watershed_segmentation(image.copy(), gray)
    display_image('Watershed Segmentation', watershed_result)

    # Clustering-Based Segmentation
    kmeans_result = kmeans_segmentation(image)
    display_image('K-means Segmentation', kmeans_result)

    # Deep Learning-Based Segmentation
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Note: Loading a dataset and training the model is required here
    # Uncomment the following line after preparing the dataset
    # model.fit(train_images, train_masks, epochs=10, validation_data=(val_images, val_masks))

    print("Segmentation techniques have been applied and displayed.")
