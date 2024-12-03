import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_integral_image(img):
    return cv2.integral(img)

# Two-rectangle Haar feature (horizontal)
def haar_feature_two_horizontal(integral_img, x, y, width, height):
    
    left_sum = integral_img[y + height, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height, x] + integral_img[y, x]
    
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + width // 2] + integral_img[y, x + width // 2]
    
    return left_sum - right_sum




# Two-rectangle Haar feature (vertical)
def haar_feature_two_vertical(integral_img, x, y, width, height): # eyes
    top_sum = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
              - integral_img[y + height // 2, x] + integral_img[y, x]
    bottom_sum = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                 - integral_img[y + height, x] + integral_img[y + height // 2, x]
    return top_sum - bottom_sum

# Three-rectangle Haar feature
def haar_feature_three_horizontal(integral_img, x, y, width, height): # nose bridge
    left_sum = integral_img[y + height, x + width // 3] - integral_img[y, x + width // 3] \
               - integral_img[y + height, x] + integral_img[y, x]
    middle_sum = integral_img[y + height, x + 2 * width // 3] - integral_img[y, x + 2 * width // 3] \
                 - integral_img[y + height, x + width // 3] + integral_img[y, x + width // 3]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + 2 * width // 3] + integral_img[y, x + 2 * width // 3]
    return left_sum - 2 * middle_sum + right_sum

# Four-rectangle Haar feature
def haar_feature_four_rectangle(integral_img, x, y, width, height): # 
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
    plt.figure
    plt.imshow(integral_img, cmap='gray')
    plt.show()
    detected_faces = []

    # Sliding window approach
    for y in range(0, image.shape[0] - window_size[1], 5):  # Step size of 4 pixels
        for x in range(0, image.shape[1] - window_size[0], 5):  # Step size of 4 pixels
            # Extract multiple Haar features
            f1 = haar_feature_two_horizontal(integral_img, x, y, window_size[0], window_size[1])
            #f2 = haar_feature_two_vertical(integral_img, x, y, window_size[0], window_size[1])
            #f3 = haar_feature_three_horizontal(integral_img, x, y, window_size[0], window_size[1])
            #f4 = haar_feature_four_rectangle(integral_img, x, y, window_size[0], window_size[1])

            # Combine features with voting mechanism
            score = 0
            if f1 > thresholds[0] and f1 < thresholds[1]: 
                score += 1
                print(f1)
            if f2 > thresholds[0] and f2 < thresholds[1]: 
                score += 1
                print(f2)
            if f3 > thresholds[0] and f3 < thresholds[1]: 
                score += 1
                print(f3)
            if f4 > thresholds[0] and f4 < thresholds[1]: 
                score += 1

            # If enough features vote for a face, mark it as detected
            if score >= 1:
                detected_faces.append((x, y, window_size[0], window_size[1]))

    return detected_faces

images = []

# Loop through image indices (adjust the range if needed)
for i in range(0, 1522):  # Assuming the filenames start from 0000.pgm to 1521.pgm
    image_path = f'face-database/BioID_{i:04}.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(image,(5,5),10)
    
    # Check if the image was loaded successfully
    if blur is None:
        print(f"Warning: Could not load image at {image_path}")
    else:
        images.append(blur)
        window_size = (100, 100)  # Size of the sliding window
        thresholds = [99500, 100000, 600000, 600000]  # Thresholds for each feature (adjust as needed)

        # Detect faces in the image
        faces = detect_faces(blur, window_size, thresholds)

        # Draw detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(blur, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        plt.imshow(blur, cmap='gray')
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
     
     



    




# # Create a sample image
# image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.double)
# image = image/6
# img1 = compute_integral_img(image)
# print(img1)
# plt.figure(1)
# plt.imshow(img1,cmap='gray')


# # Calculate the integral image
# integral_image = cv2.integral(image)
# plt.figure(2)
# plt.imshow(integral_image, cmap='gray')
# plt.show()
# print(integral_image)

# images = []
# num = 1
# for i in [1,17,29]:  
#     image_path = f'face-database/BioID_{i:04}.pgm'
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = image/np.max(image) 
#     plt.figure(num)
#     num +=1
#     plt.imshow(image, cmap='gray')

#     # Calculate the integral image
#     integral_image = cv2.integral(image)
#     print(integral_image)
#     plt.figure(num)
#     num +=1
#     plt.imshow(integral_image, cmap='gray')

# plt.show()