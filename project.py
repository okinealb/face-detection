import numpy as np
from skimage.feature import haar_like_feature

# Create a sample image
image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

# Calculate a 2-rectangle Haar feature
feature_value = haar_like_feature(image, 0, 0, 2, 2, 'type-2-x')

print(feature_value)
