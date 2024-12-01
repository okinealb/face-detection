import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the integral image
# Function to compute the integral image
def compute_integral_image(image):
    return cv2.integral(image)

# Two-rectangle Haar feature (horizontal)
def haar_feature_two_horizontal(integral_img, x, y, width, height):
    left_sum = integral_img[y + height, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height, x] + integral_img[y, x]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + width // 2] + integral_img[y, x + width // 2]
    return left_sum - right_sum

# Two-rectangle Haar feature (vertical)
def haar_feature_two_vertical(integral_img, x, y, width, height):
    top_sum = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
              - integral_img[y + height // 2, x] + integral_img[y, x]
    bottom_sum = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                 - integral_img[y + height, x] + integral_img[y + height // 2, x]
    return top_sum - bottom_sum

# Three-rectangle Haar feature
def haar_feature_three_horizontal(integral_img, x, y, width, height):
    left_sum = integral_img[y + height, x + width // 3] - integral_img[y, x + width // 3] \
               - integral_img[y + height, x] + integral_img[y, x]
    middle_sum = integral_img[y + height, x + 2 * width // 3] - integral_img[y, x + 2 * width // 3] \
                 - integral_img[y + height, x + width // 3] + integral_img[y, x + width // 3]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + 2 * width // 3] + integral_img[y, x + 2 * width // 3]
    return left_sum - 2 * middle_sum + right_sum

# Four-rectangle Haar feature
def haar_feature_four_rectangle(integral_img, x, y, width, height):
    top_left = integral_img[y + height // 2, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height // 2, x] + integral_img[y, x]
    top_right = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
                - integral_img[y + height // 2, x + width // 2] + integral_img[y, x + width // 2]
    bottom_left = integral_img[y + height, x + width // 2] - integral_img[y + height // 2, x + width // 2] \
                  - integral_img[y + height, x] + integral_img[y + height // 2, x]
    bottom_right = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                   - integral_img[y + height, x + width // 2] + integral_img[y + height // 2, x + width // 2]

    return top_left + bottom_right - top_right - bottom_left

# Function for simple face detection using combined Haar features
def detect_faces(image, window_size, thresholds):
    integral_img = compute_integral_image(image)
    detected_faces = []

    # Sliding window approach
    for y in range(0, image.shape[0] - window_size[1], 4):  # Step size of 4 pixels
        for x in range(0, image.shape[1] - window_size[0], 4):  # Step size of 4 pixels
            # Extract multiple Haar features
            f1 = haar_feature_two_horizontal(integral_img, x, y, window_size[0], window_size[1])
            f2 = haar_feature_two_vertical(integral_img, x, y, window_size[0], window_size[1])
            f3 = haar_feature_three_horizontal(integral_img, x, y, window_size[0], window_size[1])
            f4 = haar_feature_four_rectangle(integral_img, x, y, window_size[0], window_size[1])

            # Combine features with a simple voting mechanism
            score = 0
            if f1 > thresholds[0]: score += 1
            if f2 > thresholds[1]: score += 1
            if f3 > thresholds[2]: score += 1
            if f4 > thresholds[3]: score += 1

            # If enough features vote for a face, mark it as detected
            if score >= 3:
                detected_faces.append((x, y, window_size[0], window_size[1]))

    return detected_faces

images = []

# Loop through image indices (adjust the range if needed)
for i in range(0, 1):  # Assuming the filenames start from 0000.pgm to 1521.pgm
    image_path = f'face-database/BioID_{i:04}.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: Could not load image at {image_path}")
    else:
        images.append(image)
        window_size = (100, 100)  # Size of the sliding window
        thresholds = [200000, 200000, 200000, 200000]  # Thresholds for each feature (adjust as needed)

        # Detect faces in the image
        faces = detect_faces(image, window_size, thresholds)

        # Draw detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        plt.imshow(image, cmap='gray')
        plt.title('Detected Faces')
        plt.axis('off')
        plt.show()
        
        
    #         # Parameters
    #     window_size = (40, 40)  # Size of the sliding window
    #     threshold = 150000  # Feature threshold for face detection (adjust as needed)

    # # Visualize Haar feature
    #     x, y = 100, 100  # Example coordinates for visualization
    #     visualize_haar_feature(image, x, y, window_size[0], window_size[1])

    # # Detect faces in the image
    #     faces = detect_faces(image, window_size, threshold)

    # # Draw detected faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # Display the result
    #     plt.imshow(image, cmap='gray')
    #     plt.title('Detected Faces')
    #     plt.axis('off')
    #     plt.show()
     
     

