# This file was our initial final test for haar extraction and then face detection.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import random

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
                - integral_img[y + height, x + 2 * width // 3] + integral_img[y, x + width // 3]
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


# class HaarCascadeTrainer:
#     def __init__(self, positive_samples, negative_samples, num_stages, min_detection_rate, max_false_positive_rate, scales=[1.0], min_improvement=0.005):
#         self.positive_samples = positive_samples
#         self.negative_samples = negative_samples
#         self.num_stages = num_stages
#         self.min_detection_rate = min_detection_rate
#         self.max_false_positive_rate = max_false_positive_rate
#         self.scales = scales
#         self.stages = []
#         self.min_improvement = min_improvement

#     def generate_multi_scale_samples(self, images):
#         """Resize images to multiple scales."""
#         multi_scale_images = []
#         for img in images:
#             for scale in self.scales:
#                 resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#                 multi_scale_images.append(resized_img)
#         return multi_scale_images

#     def compute_haar_features_parallel(self, integral_images, window_sizes=[(24, 24)]):
#         def extract_features(integral_img):
#             h, w = integral_img.shape
#             image_features = []
#             for window_size in window_sizes:
#                 ws_w, ws_h = window_size
#                 for y in range(0, h - ws_h, 4):
#                     for x in range(0, w - ws_w, 4):
#                         features = [
#                             haar_feature_two_horizontal(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_two_vertical(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_three_horizontal(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_four_rectangle(integral_img, x, y, ws_w, ws_h),
#                         ]
#                         image_features.append(features)
#             return np.concatenate(image_features).flatten() if image_features else None

#         # Run feature extraction in parallel
#         all_features = Parallel(n_jobs=-1)(delayed(extract_features)(img) for img in integral_images)
#         return [f for f in all_features if f is not None]

#     def train_weak_classifier(self, features, labels, weights):
#         num_features = features.shape[1]
#         best_error = float("inf")
#         best_classifier = None

#         for feature_idx in range(num_features):
#             feature_values = features[:, feature_idx]
#             threshold = np.percentile(feature_values, 50)
#             predictions = (feature_values >= threshold).astype(int)
#             error = np.sum(weights * (predictions != labels))
#             if error < best_error:
#                 best_error = error
#                 alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
#                 best_classifier = (feature_idx, threshold, alpha)
#         return best_classifier

#     def train_stage(self, pos_samples, neg_samples, weights, max_weak_classifiers=50):
#         print("\n--> Starting feature computation for positive and negative samples...")
        
#         pos_features = self.compute_haar_features_parallel(pos_samples)
#         neg_features = self.compute_haar_features_parallel(neg_samples)
#         print(f"--> Feature computation completed: {len(pos_features)} positives, {len(neg_features)} negatives.")

#         # Dynamic feature alignment
#         min_length = min(min(len(f) for f in pos_features), min(len(f) for f in neg_features))
#         pos_features = [f[:min_length] for f in pos_features]
#         neg_features = [f[:min_length] for f in neg_features]

#         features = np.vstack([pos_features, neg_features])
#         labels = np.array([1] * len(pos_features) + [0] * len(neg_features))

#         adjusted_weights = np.hstack([weights[:len(pos_features)], weights[len(pos_features):]])
#         adjusted_weights /= np.sum(adjusted_weights)

#         classifiers = []
#         stage_detection_rate, stage_false_positive_rate = 0, 1

#         while (
#             stage_detection_rate < self.min_detection_rate
#             and stage_false_positive_rate > self.max_false_positive_rate
#             and len(classifiers) < max_weak_classifiers
#         ):
#             weak_classifier = self.train_weak_classifier(features, labels, adjusted_weights)
#             feature_idx, threshold, alpha = weak_classifier

#             predictions = (features[:, feature_idx] >= threshold).astype(int)
#             adjusted_weights *= np.exp(-alpha * labels * (2 * predictions - 1))
#             adjusted_weights /= np.sum(adjusted_weights)

#             classifiers.append(weak_classifier)
#             stage_detection_rate = np.mean(predictions[:len(pos_features)] == labels[:len(pos_features)])
#             stage_false_positive_rate = np.mean(predictions[len(pos_features):] != labels[len(pos_features):])

#             print(f"   Trained weak classifier {len(classifiers)}: "
#                   f"Detection Rate = {stage_detection_rate:.4f}, "
#                   f"False Positive Rate = {stage_false_positive_rate:.4f}")
        
#         print(f"--> Stage completed with {len(classifiers)} weak classifiers.")
#         return classifiers

#     def train(self):
#         """Train the cascade classifier."""
#         pos_scaled = self.generate_multi_scale_samples(self.positive_samples)
#         neg_scaled = self.generate_multi_scale_samples(self.negative_samples)

#         pos_integral_images = [compute_integral_image(img) for img in pos_scaled]
#         neg_integral_images = [compute_integral_image(img) for img in neg_scaled]

#         weights = np.hstack([np.ones(len(pos_integral_images)) / len(pos_integral_images),
#                              np.ones(len(neg_integral_images)) / len(neg_integral_images)])

#         for stage_idx in range(self.num_stages):
#             print(f"\n=== Training Stage {stage_idx + 1}/{self.num_stages} ===")
#             stage_classifiers = self.train_stage(pos_integral_images, neg_integral_images, weights)
#             print(f"Stage {stage_idx + 1} trained successfully with {len(stage_classifiers)} weak classifiers.")
#             self.stages.append(stage_classifiers)

#         print("\nTraining completed successfully.")

# positive_samples = []
# for i in range(0, 3000):
#     positive = cv2.imread(f'lfwcropped/img_{i:04}.jpg', cv2.IMREAD_GRAYSCALE)
#     positive = cv2.resize(positive, (24,24))
#     positive_samples.append(positive)
# print(f"Size of Each Image is: {positive_samples[0].shape}")
# print(f"Generated {len(positive_samples)} positive samples.")


# negative_samples = []
# for i in range(0,5000):
#     # Read in the image
#     negative_img = cv2.imread(f'databases/negatives_final/img_{i:04}.jpg', cv2.IMREAD_GRAYSCALE)
#     # Extract the subregion of the same window size from the image
#     for j in range(20):
#         y = random.randint(0, 226)
#         x = random.randint(0, 226)
#         negative_reg = negative_img[y:y + 24, x:x + 24]
#         negative_samples.append(negative_reg)
# print(f"Generated {len(negative_samples)} negative samples.")

# scales = [1.0]
# trainer = HaarCascadeTrainer(positive_samples, negative_samples, num_stages=10, 
#                              min_detection_rate=0.90, max_false_positive_rate=0.10, scales = scales)
# num_positive = len(positive_samples)
# # trainer.train(neg_sample_limit=num_positive)
# trainer.train()
# # Save the trained model
# with open("haar_cascade_model_2.pkl", "wb") as model_file:
#     pickle.dump(trainer.stages, model_file)

# print("Model saved successfully!")

# # Save the trained model stages and feature map during training
# index_map = {
#     i: "haar_feature_two_horizontal" if i % 4 == 0 else
#        "haar_feature_two_vertical" if i % 4 == 1 else
#        "haar_feature_three_horizontal" if i % 4 == 2 else
#        "haar_feature_four_rectangle"
#     for i in range(25000)  # Adjust range to include all possible indices
# }

# # Save the index map for dynamic use during detection
# with open("index_map.pkl", "wb") as map_file:
#     pickle.dump(index_map, map_file)

# class HaarCascadeTrainer:
#     def __init__(self, positive_samples, negative_samples, num_stages, min_detection_rate, max_false_positive_rate, scales=[1.0], min_improvement=0.005):
#         self.positive_samples = positive_samples
#         self.negative_samples = negative_samples
#         self.num_stages = num_stages
#         self.min_detection_rate = min_detection_rate
#         self.max_false_positive_rate = max_false_positive_rate
#         self.scales = scales
#         self.stages = []
#         self.min_improvement = min_improvement

#     def generate_multi_scale_samples(self, images):
#         """Resize images to multiple scales."""
#         multi_scale_images = []
#         for img in images:
#             for scale in self.scales:
#                 resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#                 multi_scale_images.append(resized_img)
#         return multi_scale_images

#     def compute_haar_features_parallel(self, integral_images, window_sizes=[(100, 100)]):
#         def extract_features(integral_img):
#             h, w = integral_img.shape
#             image_features = []
#             for window_size in window_sizes:
#                 ws_w, ws_h = window_size
#                 for y in range(0, h - ws_h, 8):  # Larger step size
#                     for x in range(0, w - ws_w, 8):
#                         features = [
#                             haar_feature_two_horizontal(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_two_vertical(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_three_horizontal(integral_img, x, y, ws_w, ws_h),
#                             haar_feature_four_rectangle(integral_img, x, y, ws_w, ws_h),
#                         ]
#                         image_features.append(features)
#             return np.concatenate(image_features).flatten() if image_features else None

#         # Run feature extraction in parallel
#         all_features = Parallel(n_jobs=-1)(delayed(extract_features)(img) for img in integral_images)
#         return [f for f in all_features if f is not None]

#     def pad_or_truncate(self, features, target_length):
#         """Pad or truncate feature vectors to a consistent length."""
#         padded_features = []
#         for f in features:
#             if len(f) > target_length:
#                 padded_features.append(f[:target_length])
#             else:
#                 padded_features.append(np.pad(f, (0, target_length - len(f)), mode='constant'))
#         return np.array(padded_features)

#     def train_weak_classifier(self, features, labels, weights):
#         num_features = features.shape[1]
#         best_error = float("inf")
#         best_classifier = None

#         for feature_idx in range(num_features):
#             feature_values = features[:, feature_idx]
#             threshold = np.median(feature_values)
#             predictions = (feature_values >= threshold).astype(int)
#             error = np.sum(weights * (predictions != labels))

#             # Debugging: Print the error
#             print(f"Feature {feature_idx}: Error = {error}, Threshold = {threshold}")

#             if error < best_error:
#                 best_error = error
#                 alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))  # Avoid division by zero
#                 best_classifier = (feature_idx, threshold, alpha)
#         return best_classifier

    
#     def train_stage(self, pos_samples, neg_samples, weights, neg_sample_limit, max_weak_classifiers=1000):
#         print("\n--> Starting feature computation for positive and negative samples...")
#         sampled_neg_indices = np.random.choice(len(neg_samples), neg_sample_limit, replace=False)
#         sampled_neg_samples = [neg_samples[i] for i in sampled_neg_indices]

#         pos_features = self.compute_haar_features_parallel(pos_samples)
#         neg_features = self.compute_haar_features_parallel(sampled_neg_samples)
#         print(f"--> Feature computation completed: {len(pos_features)} positives, {len(neg_features)} negatives.")

#         min_length = min(min(len(f) for f in pos_features), min(len(f) for f in neg_features))
#         pos_features = self.pad_or_truncate(pos_features, min_length)
#         neg_features = self.pad_or_truncate(neg_features, min_length)

#         features = np.vstack([pos_features, neg_features])
#         labels = np.array([1] * len(pos_features) + [0] * len(neg_features))

#         pos_weights = weights[:len(pos_samples)][:len(pos_features)]  # Match weights to features
#         neg_weights = weights[len(pos_samples):][sampled_neg_indices]
#         neg_weights = neg_weights[:len(neg_features)]  # Ensure consistency in length
#         adjusted_weights = np.hstack([pos_weights, neg_weights])

#         # Normalize weights
#         adjusted_weights /= np.sum(adjusted_weights)

#         print(f"Features shape: {features.shape}")
#         print(f"Labels length: {len(labels)}")
#         print(f"Weights length: {len(adjusted_weights)}")

#         if len(adjusted_weights) != features.shape[0]:
#             print(f"Adjusted weights: {len(adjusted_weights)}, Features: {features.shape[0]}")
#             raise ValueError("Mismatch in weights and features length.")

#         classifiers = []
#         stage_detection_rate, stage_false_positive_rate = 0, 1

#         while (
#             stage_detection_rate < self.min_detection_rate
#             and stage_false_positive_rate > self.max_false_positive_rate
#             and len(classifiers) < max_weak_classifiers
#         ):
#             weak_classifier = self.train_weak_classifier(features, labels, adjusted_weights)
#             feature_idx, threshold, alpha = weak_classifier

#             predictions = (features[:, feature_idx] >= threshold).astype(int)
#             adjusted_weights *= np.exp(-alpha * labels * (2 * predictions - 1))
#             adjusted_weights /= np.sum(adjusted_weights)

#             classifiers.append(weak_classifier)
#             stage_detection_rate = np.mean(predictions[:len(pos_features)] == labels[:len(pos_features)])
#             stage_false_positive_rate = np.mean(predictions[len(pos_features):] != labels[len(pos_features):])

#             print(f"   Trained weak classifier {len(classifiers)}: "
#                 f"Detection Rate = {stage_detection_rate:.4f}, "
#                 f"False Positive Rate = {stage_false_positive_rate:.4f}")

#         print(f"--> Stage completed with {len(classifiers)} weak classifiers.")
#         return classifiers



#     # Hard negative mining updated
#     def detect_false_positives_parallel(self, negative_samples, current_stages, max_hard_negatives=30):
#         """Detect hard negatives with a limit on the number of mined negatives."""
#         def process_image(idx, img):
#             hard_negatives = []
#             integral_img = compute_integral_image(img)
#             h, w = img.shape
#             for y in range(0, h - 100, 12):
#                 for x in range(0, w - 100, 12):
#                     is_face = True
#                     for stage in current_stages:
#                         stage_score = 0
#                         for feature_idx, threshold, alpha in stage:
#                             feature_value = haar_feature_two_horizontal(integral_img, x, y, 100, 100)
#                             prediction = 1 if feature_value >= threshold else 0
#                             stage_score += alpha * (1 if prediction == 1 else -1)
#                         if stage_score < 0:
#                             is_face = False
#                             break
#                     if is_face:
#                         hard_negatives.append(img[y:y+100, x:x+100])
#                         if len(hard_negatives) >= max_hard_negatives // len(negative_samples):
#                             return hard_negatives
#             return hard_negatives

#         print("--> Hard Negative Mining in progress...")
#         hard_negatives = Parallel(n_jobs=-1)(delayed(process_image)(idx, img) 
#                                             for idx, img in enumerate(negative_samples))
#         hard_negatives = [hn for sublist in hard_negatives for hn in sublist][:max_hard_negatives]
#         print(f"--> Hard Negative Mining completed: {len(hard_negatives)} new negatives collected.")
#         return hard_negatives

#     # Training function remains the same except for hard negative limits
#     def train(self, neg_sample_limit):
#         """Train the cascade classifier with hard negative mining."""
#         pos_scaled = self.generate_multi_scale_samples(self.positive_samples)
#         neg_scaled = self.generate_multi_scale_samples(self.negative_samples)

#         pos_integral_images = [compute_integral_image(img) for img in pos_scaled]
#         neg_integral_images = [compute_integral_image(img) for img in neg_scaled]

#         # Initialize weights for positive and negative samples
#         weights = np.hstack([np.ones(len(pos_integral_images)) / len(pos_integral_images),
#                             np.ones(len(neg_integral_images)) / len(neg_integral_images)])

#         for stage_idx in range(self.num_stages):
#             print(f"\n=== Training Stage {stage_idx + 1}/{self.num_stages} ===")

#             # Train the stage
#             stage_classifiers = self.train_stage(pos_integral_images, neg_integral_images, weights, neg_sample_limit)
#             print(f"Stage {stage_idx + 1} trained successfully with {len(stage_classifiers)} weak classifiers.")

#             self.stages.append(stage_classifiers)

#             # Hard Negative Mining
#             print("Starting hard negative mining...")
#             hard_negatives = self.detect_false_positives_parallel(neg_scaled, self.stages)
#             print(f"Stage {stage_idx + 1}: Collected {len(hard_negatives)} hard negatives.")

#             if hard_negatives:
#                 # Update the list of negative samples
#                 neg_scaled.extend(hard_negatives)
#                 neg_integral_images.extend([compute_integral_image(img) for img in hard_negatives])

#                 # Extend weights for new negatives
#                 hard_negative_weights = np.ones(len(hard_negatives)) / (2 * len(hard_negatives))
#                 weights = np.hstack([weights, hard_negative_weights])

#                 # Normalize weights to sum to 1
#                 weights /= np.sum(weights)

#             print(f"Stage {stage_idx + 1} completed. Total negatives now: {len(neg_scaled)}")

#         print("\nTraining completed successfully.")

# Training stops HERE! - Useful for commenting and uncommenting.


# Load the model and index map
with open("haar_cascade_model_2.pkl", "rb") as model_file:
    loaded_stages = pickle.load(model_file)
with open("index_map.pkl", "rb") as map_file:
    index_map = pickle.load(map_file)

print("Model and index map loaded successfully!")
print("Loaded stages:")
for i, stage in enumerate(loaded_stages):
    print(f"Stage {i}: {stage}")

def combine_bounding_boxes(boxes):
    """
    Combine multiple bounding boxes into a single bounding box.
    :param boxes: List of bounding boxes [x, y, w, h].
    :return: A single bounding box [x_min, y_min, width, height].
    """
    if len(boxes) == 0:
        return None

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)

    # Calculate new width and height
    combined_width = x_max - x_min
    combined_height = y_max - y_min

    return [x_min, y_min, combined_width, combined_height]
def non_maximum_suppression(boxes, scores, overlap_thresh=0.4):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes based on their scores.
    :param boxes: List of bounding boxes [x, y, w, h].
    :param scores: List of scores corresponding to each box.
    :param overlap_thresh: IoU threshold for merging boxes.
    :return: Filtered list of bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Convert to float for precision
    boxes = np.array(boxes, dtype="float")
    scores = np.array(scores)

    # Coordinates of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]  # Sort by scores descending

    picked_boxes = []

    while len(indices) > 0:
        i = indices[0]
        picked_boxes.append(boxes[i].astype("int"))

        # Compute IoU with other boxes
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[indices[1:]]

        # Remove indices where IoU > overlap_thresh
        indices = indices[np.where(overlap <= overlap_thresh)[0] + 1]

    return picked_boxes


def resize_window_to_original(window, original_width, original_height, resized_width=29, resized_height=33):
    """
    Resize a detection window from resized dimensions back to the original image dimensions.
    
    :param window: List or tuple of detection window (x, y, w, h).
    :param original_width: Width of the original image.
    :param original_height: Height of the original image.
    :param resized_width: Width of the resized image (default is 30).
    :param resized_height: Height of the resized image (default is 30).
    :return: Resized detection window (x, y, w, h) in the original image dimensions.
    """
    scaling_factor_x = original_width / resized_width
    scaling_factor_y = original_height / resized_height
    
    x, y, w, h = window
    x = int(x * scaling_factor_x)
    y = int(y * scaling_factor_y)
    w = int(w * scaling_factor_x)
    h = int(h * scaling_factor_x)
    
    return x, y, w, h
def pick_central_box(boxes, image_shape):
    """
    Pick the bounding box closest to the center of the image.
    :param boxes: List of bounding boxes [x, y, w, h].
    :param image_shape: Shape of the image (height, width).
    :return: Box closest to the center.
    """
    if len(boxes) == 0:
        return None

    center_x = image_shape[1] // 2
    center_y = image_shape[0] // 2

    distances = [
        ((box[0] + box[2] // 2 - center_x) ** 2 + (box[1] + box[3] // 2 - center_y) ** 2)
        for box in boxes
    ]

    return boxes[np.argmin(distances)]
def detect_faces(image, cascade_stages, window_size=(150, 150), step_ratio=0.05, nms_threshold=0.4, confidence_threshold=1.0):
    integral_img = compute_integral_image(image)
    detected_boxes = []
    box_scores = []

    step_size = int(image.shape[1] * step_ratio)

    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            is_face = True
            stage_score = 0

            for stage in cascade_stages:
                for feature_idx, threshold, alpha in stage:
                    feature_name = index_map.get(feature_idx)
                    if feature_name == "haar_feature_two_horizontal":
                        feature_value = haar_feature_two_horizontal(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_two_vertical":
                        feature_value = haar_feature_two_vertical(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_three_horizontal":
                        feature_value = haar_feature_three_horizontal(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_four_rectangle":
                        feature_value = haar_feature_four_rectangle(integral_img, x, y, *window_size)
                    else:
                        raise ValueError(f"Unknown feature: {feature_name}")

                    prediction = 1 if feature_value >= threshold else 0
                    stage_score += alpha * (1 if prediction == 1 else -1)

                if stage_score < 0:  # Reject the window if it fails any stage
                    is_face = False
                    break

            if is_face and stage_score >= confidence_threshold:
                detected_boxes.append([x, y, window_size[0], window_size[1]])
                box_scores.append(stage_score)

    nms_boxes = non_maximum_suppression(detected_boxes, box_scores, overlap_thresh=nms_threshold)
    if len(nms_boxes) > 1:
        nms_boxes = [pick_central_box(nms_boxes, image.shape)]
    return nms_boxes

def detect_faces(image, cascade_stages, window_size=(24, 24), step_ratio=0.05, nms_threshold=0.4, confidence_threshold=1.0):
    """
    Detect faces in an image and return bounding boxes for all detected faces, filtered by confidence score.
    :param image: Input grayscale image.
    :param cascade_stages: Loaded Haar Cascade model.
    :param window_size: Size of the detection window.
    :param step_ratio: Step size as a fraction of window size.
    :param nms_threshold: IoU threshold for NMS.
    :param confidence_threshold: Minimum confidence score to keep a detection.
    :return: Filtered bounding boxes after NMS and confidence filtering.
    """
    
    integral_img = compute_integral_image(image)
    detected_boxes = []
    box_scores = []

    step_size = int(image.shape[1] * step_ratio)

    # Sliding window detection
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            is_face = True
            stage_score = 0

            for stage in cascade_stages:
                for feature_idx, threshold, alpha in stage:
                    feature_name = index_map.get(feature_idx)
                    if feature_name == "haar_feature_two_horizontal":
                        feature_value = haar_feature_two_horizontal(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_two_vertical":
                        feature_value = haar_feature_two_vertical(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_three_horizontal":
                        feature_value = haar_feature_three_horizontal(integral_img, x, y, *window_size)
                    elif feature_name == "haar_feature_four_rectangle":
                        feature_value = haar_feature_four_rectangle(integral_img, x, y, *window_size)
                    else:
                        raise ValueError(f"Unknown feature: {feature_name}")

                    prediction = 1 if feature_value >= threshold else 0
                    stage_score += alpha * (1 if prediction == 1 else -1)

                if stage_score < 0:  # Reject the window if it fails any stage
                    is_face = False
                    break
                
            if is_face and stage_score >= confidence_threshold:  # Filter by confidence threshold
                detected_boxes.append([x, y, window_size[0], window_size[1]])
                box_scores.append(stage_score)

    # Apply Non-Maximum Suppression
    nms_boxes = non_maximum_suppression(detected_boxes, box_scores, overlap_thresh=nms_threshold)

    return nms_boxes

# Detect faces in good detection images
good_detection_images = []
c = 0
for i in range(0, 0):  
    image_path = f'lfwimages/img_{i:04}.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    good_detection_images.append(img)
    detected_boxes = detect_faces(good_detection_images[c], loaded_stages)

    # Draw detections
    for (x, y, w, h) in detected_boxes:
        cv2.rectangle(good_detection_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(good_detection_images[c], cmap="gray")
    plt.title("Face Features Detected")
    plt.axis("off")
    plt.savefig('milestone_4/results/' + f'Good_face_lfw_{c:03}', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving
    c += 1


# image_path = f'stockimg.jpg'
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (250,250))
# detected_boxes = detect_faces(img, loaded_stages)

# # Draw detections
# for (x, y, w, h) in detected_boxes:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# plt.imshow(img, cmap="gray")
# plt.title("Face Features Detected")
# plt.axis("off")
# plt.show()

# # Detect faces in good detection images
# good_detection_images = []
# c = 0
# for i in range(50, 71):  
#     image_path = f'face-database/BioID_{i:04}.pgm'
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     good_detection_images.append(img)
#     detected_boxes = detect_faces(good_detection_images[c], loaded_stages)

#     # Draw detections
#     for (x, y, w, h) in detected_boxes:
#         cv2.rectangle(good_detection_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

#     plt.imshow(good_detection_images[c], cmap="gray")
#     plt.title("Face Features Detected")
#     plt.axis("off")
#     plt.savefig('milestone_4/results/' + f'Good_face_{c:03}', bbox_inches='tight', pad_inches=0)
#     plt.close()  # Close the figure after saving
#     c += 1

# Detect faces in terrible face images
terrible_face_images = []
c = 0
for i in range(0, 10):  
    image_path = f'face-database/BioID_{i:04}.pgm'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    terrible_face_images.append(img)
    detected_boxes = detect_faces(terrible_face_images[c], loaded_stages)

    # Draw detections
    for (x, y, w, h) in detected_boxes:
        cv2.rectangle(terrible_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(terrible_face_images[c], cmap="gray")
    plt.title("Face Features Detected With False Positives")
    plt.axis("off")
    plt.savefig('milestone_4/results/' + f'Terrible_face_{c:03}', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving
    c += 1

# # Detect faces in non-face images
# non_face_images = []
# c = 0
# i = 41
# for j in ['ant', 'airplanes', 'beaver', 'accordion', 'bonsai', 'brain', 'brontosaurus', 'cellphone',  'camera', 'beaver', 'ceiling_fan', 
#           'ferry', 'flamingo', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
#           'crab', 'dolphin', 'electric_guitar', 'lamp', 'Motorbikes']:  
#     image_path = f'databases/caltech-101/{j}/image_{i:04}.jpg'
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     non_face_images.append(img)
#     detected_boxes = detect_faces(non_face_images[c], loaded_stages)

#     # Draw detections
#     for (x, y, w, h) in detected_boxes:
#         cv2.rectangle(non_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

#     plt.imshow(non_face_images[c], cmap="gray")
#     plt.title("Non-Face Image Features Detected")
#     plt.axis("off")
#     plt.savefig('milestone_4/results/' + f'Non_face_{c:03}', bbox_inches='tight', pad_inches=0)
#     plt.close()  # Close the figure after saving
#     c += 1
