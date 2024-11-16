import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
import cv2

# Create a sample image
image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

# Calculate a 2-rectangle Haar feature
feature_value = haar_like_feature(image, 0, 0, 2, 2, 'type-2-x')

print(feature_value)



# Create a sample image
imagetwo = np.zeros((10, 10))

# Define the Haar feature coordinates
feature_coord, _ = haar_like_feature_coord(width=10, height=10, feature_type='type-2-x')


# Plot the image and the Haar feature
plt.imshow(imagetwo, cmap='gray')

#for (x, y) in feature_coord:
#    plt.gca().add_patch(plt.Rectangle((x(0), x(1)), y(0), y(1), fill=False, edgecolor='red'))
plt.show()



# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image
imagethree = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(imagethree, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image
cv2.imshow('Faces', imagethree)
cv2.waitKey(0)