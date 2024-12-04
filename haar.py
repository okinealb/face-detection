import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


# Haar Feature Computation
def compute_integral_image(img):
    return cv2.integral(img)

def haar_feature_two_horizontal(integral_img, x, y, width, height):
    left_sum = integral_img[y + height, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height, x] + integral_img[y, x]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + width // 2] + integral_img[y, x + width // 2]
    return left_sum - right_sum

def haar_feature_two_vertical(integral_img, x, y, width, height):
    top_sum = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
              - integral_img[y + height // 2, x] + integral_img[y, x]
    bottom_sum = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                 - integral_img[y + height, x] + integral_img[y + height // 2, x]
    return top_sum - bottom_sum

def haar_feature_three_horizontal(integral_img, x, y, width, height):
    left_sum = integral_img[y + height, x + width // 3] - integral_img[y, x + width // 3] \
               - integral_img[y + height, x] + integral_img[y, x]
    middle_sum = integral_img[y + height, x + 2 * width // 3] - integral_img[y, x + 2 * width // 3] \
                 - integral_img[y + height, x + width // 3] + integral_img[y, x + width // 3]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + 2 * width // 3] + integral_img[y, x + 2 * width // 3]
    return left_sum - 2 * middle_sum + right_sum

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


# AdaBoost Cascade Training
class HaarCascadeTrainer:
    def __init__(self, positive_samples, negative_samples, num_stages, min_detection_rate, max_false_positive_rate):
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.num_stages = num_stages
        self.min_detection_rate = min_detection_rate
        self.max_false_positive_rate = max_false_positive_rate
        self.stages = []
    
    def compute_haar_features(self, integral_images):
        features = []
        for integral_img in integral_images:
            h, w = integral_img.shape
            sample_features = []
            for y in range(0, h - 24, 4):  # Slide vertically with step size 4
                for x in range(0, w - 24, 4):  # Slide horizontally with step size 4
                    # Ensure the feature window fits within the image
                    if y + 24 > h or x + 24 > w:
                        continue
                    sample_features.append([
                        haar_feature_two_horizontal(integral_img, x, y, 24, 24),
                        haar_feature_two_vertical(integral_img, x, y, 24, 24),
                        haar_feature_three_horizontal(integral_img, x, y, 24, 24),
                        haar_feature_four_rectangle(integral_img, x, y, 24, 24),
                    ])
            features.append(np.array(sample_features).flatten())  # Flatten features for each sample
            
        return np.array(features)

    
    def train_weak_classifier(self, features, labels, weights):
        """Train a weak classifier on weighted samples."""
        print(f"Features shape: {features.shape}")
        num_features = features.shape[1]  # Number of features
        best_error = float("inf")
        best_classifier = None

        for feature_idx in range(num_features):
            # Extract values for a single feature
            feature_values = features[:, feature_idx]  # Shape: (N_samples,)
            print(f"Feature {feature_idx}, Feature values shape: {feature_values.shape}, Labels shape: {labels.shape}")

            # Compute the threshold (e.g., median)
            threshold = np.median(feature_values)

            # Compute predictions for this feature
            predictions = (feature_values >= threshold).astype(int)  # Shape: (N_samples,)
            print(f"Feature values shape: {feature_values.shape}, Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")

            # Calculate the weighted error
            error = np.sum(weights * (predictions != labels))

            # Track the best weak classifier
            if error < best_error:
                best_error = error
                alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))  # Avoid division by zero
                best_classifier = (feature_idx, threshold, alpha)

        return best_classifier

    def train_stage(self, pos_samples, neg_samples, weights):
        """Train a single stage of the cascade."""
        features = np.vstack([self.compute_haar_features(pos_samples), self.compute_haar_features(neg_samples)])
        labels = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
        classifiers = []
        stage_detection_rate = 0
        stage_false_positive_rate = 1

        while stage_detection_rate < self.min_detection_rate and stage_false_positive_rate > self.max_false_positive_rate:
            weak_classifier = self.train_weak_classifier(features, labels, weights)
            feature_idx, threshold, alpha = weak_classifier

            predictions = (features[:, feature_idx] >= threshold).astype(int)
            weights *= np.exp(-alpha * labels * (2 * predictions - 1))
            weights /= np.sum(weights)

            classifiers.append(weak_classifier)
            stage_detection_rate = np.mean(predictions[:len(pos_samples)] == labels[:len(pos_samples)])
            stage_false_positive_rate = np.mean(predictions[len(pos_samples):] != labels[len(pos_samples):])

        return classifiers

    def train(self):
        """Train the cascade classifier."""
        pos_integral_images = [compute_integral_image(img) for img in self.positive_samples]
        neg_integral_images = [compute_integral_image(img) for img in self.negative_samples]
        weights = np.hstack([np.ones(len(pos_integral_images)) / len(pos_integral_images),
                             np.ones(len(neg_integral_images)) / len(neg_integral_images)])

        for stage_idx in range(self.num_stages):
            print(f"Training stage {stage_idx + 1}/{self.num_stages}...")
            stage_classifiers = self.train_stage(pos_integral_images, neg_integral_images, weights)
            self.stages.append(stage_classifiers)

        print("Training completed.")

# Example Usage
positive_samples = [cv2.imread(f'face-database/BioID_{i:04}.pgm', cv2.IMREAD_GRAYSCALE) for i in range(50)]
negative_samples = [np.random.randint(0, 255, (positive_samples[1].shape), dtype=np.uint8) for _ in range(50)]

trainer = HaarCascadeTrainer(positive_samples, negative_samples, num_stages=5, 
                             min_detection_rate=0.995, max_false_positive_rate=0.5)
trainer.train()

# Save the trained model
with open("haar_cascade_model.pkl", "wb") as model_file:
    pickle.dump(trainer.stages, model_file)

print("Model saved successfully!")


# Load the model
with open("haar_cascade_model.pkl", "rb") as model_file:
    loaded_stages = pickle.load(model_file)

print("Model loaded successfully!")

print("Loaded stages:")
for i, stage in enumerate(loaded_stages):
    print(f"Stage {i}: {stage}")

def detect_faces(image, cascade_stages, window_size=(24, 24), step_size=4):
    """
    Detect faces in an image using the trained cascade stages.
    """
    integral_img = compute_integral_image(image)
    detected_faces = []

    # Map the large feature indices to the extracted features
    index_map = {
        3272: 0,  # Maps to `haar_feature_two_horizontal`
        12008: 1,  # Maps to `haar_feature_two_vertical`
        # Add additional mappings if needed
    }

    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            features = [
                haar_feature_two_horizontal(integral_img, x, y, window_size[0], window_size[1]),
                haar_feature_two_vertical(integral_img, x, y, window_size[0], window_size[1]),
                haar_feature_three_horizontal(integral_img, x, y, window_size[0], window_size[1]),
                haar_feature_four_rectangle(integral_img, x, y, window_size[0], window_size[1]),
            ]

            is_face = True
            for stage in cascade_stages:
                print(f"Feature indices in current stage: {[idx for idx, _, _ in stage]}")
                print(f"Features length: {len(features)}")
                stage_score = 0
                for feature_idx, threshold, alpha in stage:
                    # Map the feature index
                    mapped_idx = index_map.get(feature_idx, None)
                    if mapped_idx is None or mapped_idx >= len(features):
                        raise ValueError(f"Feature index {feature_idx} is out of range or unmapped.")
                    
                    feature_value = features[mapped_idx]
                    prediction = 1 if feature_value >= threshold else 0
                    stage_score += alpha * (1 if prediction == 1 else -1)
                if stage_score < 0:  # Reject window
                    is_face = False
                    break

            if is_face:
                detected_faces.append((x, y, window_size[0], window_size[1]))

    return detected_faces


test_image = cv2.imread('face-database/BioID_0008.pgm', cv2.IMREAD_GRAYSCALE)
detected = detect_faces(test_image, loaded_stages)

# Draw detections
for (x, y, w, h) in detected:
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(test_image, cmap="gray")
plt.title("Detected Objects")
plt.axis("off")
plt.show()