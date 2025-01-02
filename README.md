# Image Segmentation by Clustering and Preprocessing

## Introduction

### What is Image Segmentation?

Image segmentation is a technique used in computer vision to partition an image into multiple segments or regions, making it easier to analyze. The goal is to simplify the representation of an image or make it more meaningful, which helps in tasks such as object detection, feature extraction, and image compression.

There are different types of image segmentation:
- **Semantic Segmentation**: Classifies each pixel in the image as belonging to a specific class (e.g., road, car). It doesn't differentiate between instances of the same class.
- **Instance Segmentation**: Differentiates between distinct objects of the same class (e.g., two different cars).
- **Panoptic Segmentation**: Combines both semantic and instance segmentation to provide a detailed image analysis.

### What is Clustering in Image Segmentation?

Clustering in image segmentation is an unsupervised learning technique where similar pixels in an image are grouped into clusters based on certain features such as color, intensity, or texture. This method is used to divide the image into meaningful regions without requiring labeled data.

### Popular Clustering Algorithms:
- **K-Means Clustering**: A widely used clustering algorithm that partitions data into `k` clusters based on color or intensity similarity.
- **Mean Shift Clustering**: Identifies modes in the feature space and is used for more complex segmentation tasks.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups pixels based on density, which can be helpful for handling noise or irregular cluster shapes.

## Methodology

This repository demonstrates the use of K-Means clustering for image segmentation and its evaluation with different values of `k` (number of clusters). The process involves the following steps:

1. **Image Preprocessing**: Converting images to a format suitable for clustering (e.g., resizing, grayscale conversion).
2. **Clustering with K-Means**: Segmenting the image into multiple regions using K-Means clustering with different values of `k`.
3. **Optimal K-value Determination**: Using the Elbow Method to identify the optimal value of `k` for better segmentation accuracy.
4. **Grayscale Preprocessing**: Converting images to grayscale before performing segmentation to simplify the clustering process.

## Tasks Performed

1. **Image Segmentation by Clustering (k=2, k=3, k=4)**:
   - These tasks demonstrate how different values of `k` affect image segmentation. The image is divided into `k` clusters, and the pixels are grouped accordingly to generate segmented images.
   - The results are displayed for `k=2`, `k=3`, and `k=4` to showcase how varying the number of clusters changes the output of the segmentation.

2. **Optimal K-value through Elbow Method**:
   - The Elbow Method is applied to determine the best value of `k` for K-Means clustering. By calculating the Within-Cluster Sum of Squares (WCSS) for various values of `k`, we identify the point where adding more clusters no longer significantly reduces the WCSS. This point, or "elbow," suggests the optimal `k` for the dataset.

3. **Grayscale Preprocessing and Segmentation**:
   - The image is converted to grayscale to simplify the clustering process. The K-Means clustering is then applied to the grayscale image to see how well the algorithm can segment the image based on pixel intensity.

## Code Implementation

The code is implemented in Python using the following libraries:
- **OpenCV** for image loading, resizing, and conversion.
- **Matplotlib** for visualization.
- **Scikit-learn** for the K-Means clustering algorithm and the Elbow Method.

### Example Code for K-Means Segmentation (k=3)
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('path_to_image.jpg')  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flatten the image
pixels = image.reshape(-1, 3)  
pixels = np.float32(pixels)  

# Perform K-Means Clustering for k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixels)
centers = np.uint8(kmeans.cluster_centers_)  
segmented_pixels = centers[labels.flatten()]  
segmented_image = segmented_pixels.reshape(image.shape)  

# Plot the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image (k=3)')
plt.axis('off')

plt.tight_layout()
plt.show()

```
![Screenshot 2024-12-18 203158](https://github.com/user-attachments/assets/cdc7c5f3-0c3d-4736-915b-728c8e00e515)

### Code for the grayscale segmentation and preprocessing of the image
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('path_to_image.jpg')  

# Convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Flatten the grayscale image
pixels = grayscale_image.flatten().reshape(-1, 1)  # Each pixel is a single intensity value
pixels = np.float32(pixels)  # Convert to float32

def segment_grayscale_image(pixels, k):
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = np.uint8(kmeans.cluster_centers_)  # Convert centers to uint8
    segmented_pixels = centers[labels.flatten()]  # Map labels to center values
    segmented_image = segmented_pixels.reshape(grayscale_image.shape)  # Reshape to original image dimensions
    return segmented_image

# Perform segmentation for k=2, 3, 4
k_values = [2, 3, 4]
segmented_images = [segment_grayscale_image(pixels, k) for k in k_values]

# Plot the original grayscale and segmented images
plt.figure(figsize=(15, 10))

# Display the grayscale image
plt.subplot(1, len(k_values) + 1, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Display the segmented images
for i, k in enumerate(k_values):
    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.imshow(segmented_images[i], cmap='gray')
    plt.title(f'Segmented (k={k})')
    plt.axis('off')

plt.tight_layout()
plt.show()
```
![Screenshot 2024-12-18 203911](https://github.com/user-attachments/assets/307149a6-82e0-450a-a9b2-c5528c949de5)



# Practical Applications of Image Segmentation by Clustering

## Practical Applications

1. **Medical Imaging**: Segment anatomical structures such as tumors or organs in MRI or CT scans.
2. **Autonomous Driving**: Segment roads, vehicles, and pedestrians for object detection and navigation.
3. **Satellite Image Analysis**: Classify land cover types (e.g., water, forest, urban) in satellite images.
4. **Content-based Image Retrieval**: Improve image search accuracy by segmenting images into meaningful regions based on their content.

## Conclusion

This project demonstrates the power of unsupervised clustering algorithms, like K-Means, for image segmentation. By adjusting the number of clusters (`k`), the segmentation output can be tailored to different image processing tasks, such as separating objects from the background or segmenting regions based on intensity or color.

Feel free to experiment with different images and adjust the parameters to see how well the clustering techniques work in various scenarios.

Under the guidance of [ Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu)
