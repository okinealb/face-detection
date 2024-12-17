import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# # AdaBoost Cascade Training
# class HaarCascadeTrainer:
#     def __init__(self, positive_samples, negative_samples, num_stages, min_detection_rate, max_false_positive_rate):
#         self.positive_samples = positive_samples
#         self.negative_samples = negative_samples
#         self.num_stages = num_stages
#         self.min_detection_rate = min_detection_rate
#         self.max_false_positive_rate = max_false_positive_rate
#         self.stages = []
    
#     def compute_haar_features(self, integral_images):
#         features = []
#         for integral_img in integral_images:
#             h, w = integral_img.shape
#             sample_features = []
#             for y in range(0, h - 24, 4):  # Slide vertically with step size 4
#                 for x in range(0, w - 24, 4):  # Slide horizontally with step size 4
#                     # Ensure the feature window fits within the image
#                     if y + 24 > h or x + 24 > w:
#                         continue
#                     sample_features.append([
#                         haar_feature_two_horizontal(integral_img, x, y, 24, 24),
#                         haar_feature_two_vertical(integral_img, x, y, 24, 24),
#                         haar_feature_three_horizontal(integral_img, x, y, 24, 24),
#                         haar_feature_four_rectangle(integral_img, x, y, 24, 24),
#                     ])
#             features.append(np.array(sample_features).flatten())  # Flatten features for each sample
            
#         return np.array(features)

    
#     def train_weak_classifier(self, features, labels, weights):
#         """Train a weak classifier on weighted samples."""
#         print(f"Features shape: {features.shape}")
#         num_features = features.shape[1]  # Number of features
#         best_error = float("inf")
#         best_classifier = None

#         for feature_idx in range(num_features):
#             # Extract values for a single feature
#             feature_values = features[:, feature_idx]  # Shape: (N_samples,)
#             print(f"Feature {feature_idx}, Feature values shape: {feature_values.shape}, Labels shape: {labels.shape}")

#             # Compute the threshold (e.g., median)
#             threshold = np.median(feature_values)

#             # Compute predictions for this feature
#             predictions = (feature_values >= threshold).astype(int)  # Shape: (N_samples,)
#             print(f"Feature values shape: {feature_values.shape}, Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")

#             # Calculate the weighted error
#             error = np.sum(weights * (predictions != labels))

#             # Track the best weak classifier
#             if error < best_error:
#                 best_error = error
#                 alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))  # Avoid division by zero
#                 best_classifier = (feature_idx, threshold, alpha)

#         return best_classifier

#     def train_stage(self, pos_samples, neg_samples, weights, neg_sample_limit):
#         """Train a single stage of the cascade with limited negative samples."""
#         # Randomly sample a subset of negative samples
#         sampled_neg_indices = np.random.choice(len(neg_samples), neg_sample_limit, replace=False)
#         sampled_neg_samples = [neg_samples[i] for i in sampled_neg_indices]

#         # Combine positive and sampled negative samples
#         features = np.vstack([self.compute_haar_features(pos_samples),
#                             self.compute_haar_features(sampled_neg_samples)])
#         labels = np.array([1] * len(pos_samples) + [0] * len(sampled_neg_samples))

#         # Adjust weights for the sampled negative samples
#         pos_weights = weights[:len(pos_samples)]
#         neg_weights = weights[len(pos_samples):][sampled_neg_indices]
#         adjusted_weights = np.hstack([pos_weights, neg_weights])

#         classifiers = []
#         stage_detection_rate = 0
#         stage_false_positive_rate = 1

#         while stage_detection_rate < self.min_detection_rate and stage_false_positive_rate > self.max_false_positive_rate:
#             weak_classifier = self.train_weak_classifier(features, labels, adjusted_weights)
#             feature_idx, threshold, alpha = weak_classifier

#             predictions = (features[:, feature_idx] >= threshold).astype(int)
#             adjusted_weights *= np.exp(-alpha * labels * (2 * predictions - 1))
#             adjusted_weights /= np.sum(adjusted_weights)

#             classifiers.append(weak_classifier)
#             stage_detection_rate = np.mean(predictions[:len(pos_samples)] == labels[:len(pos_samples)])
#             stage_false_positive_rate = np.mean(predictions[len(pos_samples):] != labels[len(pos_samples):])

#         return classifiers


#     def train(self, neg_sample_limit=100):
#         """Train the cascade classifier with negative sample limiting."""
#         pos_integral_images = [compute_integral_image(img) for img in self.positive_samples]
#         neg_integral_images = [compute_integral_image(img) for img in self.negative_samples]

#         # Initialize weights
#         weights = np.hstack([np.ones(len(pos_integral_images)) / len(pos_integral_images),
#                              np.ones(len(neg_integral_images)) / len(neg_integral_images)])

#         for stage_idx in range(self.num_stages):
#             print(f"Training stage {stage_idx + 1}/{self.num_stages}...")
#             stage_classifiers = self.train_stage(pos_integral_images, neg_integral_images, weights, neg_sample_limit)
#             self.stages.append(stage_classifiers)

#         print("Training completed.")

# # Example Usage
# # positive_samples = [cv2.imread(f'face-database/BioID_{i:04}.pgm', cv2.IMREAD_GRAYSCALE) for i in range(1521)]

# # negative_samples = [np.random.randint(0, 255, (positive_samples[1].shape), dtype=np.uint8) for _ in range(1521)]
# positive_samples = [cv2.imread(f'databases/caltech-101/Faces_easy/image_0001.jpg', cv2.IMREAD_GRAYSCALE)]
# for i in range(2, 436):
#     positive = cv2.imread(f'databases/caltech-101/Faces_easy/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE)
#     actual_positive = cv2.resize(positive, positive_samples[0].shape)
#     positive_samples.append(actual_positive)
    
# # negative = cv2.imread(f'databases/caltech-101/airplanes/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE) 
# # #     actual_negative = cv2.resize(negative, positive_samples[1].shape)
# # #     negative_samples.append(actual_negative)
    
# # positive_samples = [cv2.imread(f'databases/caltech-101/Faces_easy/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(1, 436)]
# negative_samples = [np.random.randint(0, 255, (positive_samples[1].shape), dtype=np.uint8) for _ in range(1, 1000)]

# # # for i in range(1, 800):
# # #     negative = cv2.imread(f'databases/caltech-101/airplanes/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE) 
# # #     actual_negative = cv2.resize(negative, positive_samples[1].shape)
# # #     negative_samples.append(actual_negative)
    
# for i in range(1,40):
#     for j in ['ant', 'airplanes', 'beaver', 'accordion', 'bonsai', 'brain', 'brontosaurus', 'cellphone',  'camera', 'beaver', 'ceiling_fan', 
#                'ferry', 'flamingo', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
#                'crab', 'dolphin', 'electric_guitar', 'lamp', 'Motorbikes']:
#         negative = cv2.imread(f'databases/caltech-101/{j}/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE)
#         actual_negative = cv2.resize(negative, positive_samples[1].shape)
#         negative_samples.append(actual_negative)

#     # negative_samples.append(cv2.imread(f'databases/caltech-101/bass/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/beaver/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/accordion/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/bonsai/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/brain/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/camera/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/beaver/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
#     # negative_samples.append(cv2.imread(f'databases/caltech-101/ceiling_fan/image_{i:04}.jpg', cv2.IMREAD_GRAYSCALE))
    


# trainer = HaarCascadeTrainer(positive_samples, negative_samples, num_stages=20, 
#                              min_detection_rate=0.995, max_false_positive_rate=0.3)
# trainer.train()

# # Save the trained model
# with open("haar_cascade_model_2.pkl", "wb") as model_file:
#     pickle.dump(trainer.stages, model_file)

# print("Model saved successfully!")

# # Save the trained model stages and feature map during training
# # Example of how to save feature index map
# index_map = {
#     i: "haar_feature_two_horizontal" if i % 4 == 0 else
#        "haar_feature_two_vertical" if i % 4 == 1 else
#        "haar_feature_three_horizontal" if i % 4 == 2 else
#        "haar_feature_four_rectangle"
#     for i in range(25000)  # Example: Adjust range to include all possible indices
# }

# # Save the index map for dynamic use during detection
# with open("index_map.pkl", "wb") as map_file:
#     pickle.dump(index_map, map_file)

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


def detect_faces(image, cascade_stages, window_size=(24, 24), step_size=4):
    """
    Detect faces in an image and combine features into a single bounding box.
    """
    integral_img = compute_integral_image(image)
    detected_boxes = []

    # Sliding window detection
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            is_face = True
            for stage in cascade_stages:
                stage_score = 0
                for feature_idx, threshold, alpha in stage:
                    feature_name = index_map.get(feature_idx)
                    
                    # Compute feature dynamically
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

                if stage_score < 0:  # Reject window
                    is_face = False
                    break

            if is_face:
                detected_boxes.append([x, y, window_size[0], window_size[1]])
    return detected_boxes
    # # Combine bounding boxes into one
    # final_box = combine_bounding_boxes(detected_boxes)
    # return final_box

good_detection_images = []
c = 0
for i in range(50, 71):  
    image_path = f'face-database/BioID_{i:04}.pgm'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    good_detection_images.append(img)
    detected = detect_faces(good_detection_images[c], loaded_stages)

    # # Check if a bounding box is returned
    # if detected:
    #     # Unpack the single bounding box
    #     x, y, w, h = detected
    #     cv2.rectangle(good_detection_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw detections
    for (x, y, w, h) in detected:
        cv2.rectangle(good_detection_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(good_detection_images[c], cmap="gray")
    plt.title("Face Features Detected")
    plt.axis("off")
    plt.savefig('milestone_4/results/' + f'Good_face_{c:03}', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving
    c +=1

terrible_face_images = []
c = 0
for i in range(0, 10):  
    image_path = f'face-database/BioID_{i:04}.pgm'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    terrible_face_images.append(img)
    
    
    detected = detect_faces(terrible_face_images[c], loaded_stages)

    # # Check if a bounding box is returned
    # if detected:
    #     # Unpack the single bounding box
    #     x, y, w, h = detected
    #     cv2.rectangle(terrible_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)

   
    #Draw detections
    for (x, y, w, h) in detected:
       cv2.rectangle(terrible_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(terrible_face_images[c], cmap="gray")
    plt.title("Face Features Detected With False Positives")
    plt.axis("off")
    plt.savefig('milestone_4/results/' + f'Terrible_face_{c:03}', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving
    c+=1

non_face_images = []
c = 0
i = 41
for j in ['ant', 'airplanes', 'beaver', 'accordion', 'bonsai', 'brain', 'brontosaurus', 'cellphone',  'camera', 'beaver', 'ceiling_fan', 
                'ferry', 'flamingo', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
                'crab', 'dolphin', 'electric_guitar', 'lamp', 'Motorbikes']:  
    image_path = f'databases/caltech-101/{j}/image_{i:04}.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    non_face_images.append(img)
    detected = detect_faces(non_face_images[c], loaded_stages)
        # Check if a bounding box is returned
    if detected:
        # Unpack the single bounding box
        x, y, w, h = detected
        cv2.rectangle(non_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw detections
    #for (x, y, w, h) in detected:
    #    cv2.rectangle(non_face_images[c], (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(non_face_images[c], cmap="gray")
    plt.title("Non-Face Image Features Detected")
    plt.axis("off")
    plt.savefig('milestone_4/results/' + f'Non_face_{c:03}', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving
    c += 1


# # Create montages
# def create_montage(images, tile_size, grid_cols, title):
#    num_images = len(images)
#    grid_rows = (num_images + grid_cols - 1) // grid_cols  # Calculate rows
#    montage_height = grid_rows * tile_size[1]
#    montage_width = grid_cols * tile_size[0]
#    montage = np.zeros((montage_height, montage_width), dtype=np.uint8)

#    for idx, img in enumerate(images):
#        row, col = divmod(idx, grid_cols)
#        y_start, x_start = row * tile_size[1], col * tile_size[0]
#        montage[y_start:y_start+tile_size[1], x_start:x_start+tile_size[0]] = img

#    plt.figure(figsize=(20, 10))
#    plt.imshow(montage, cmap='gray')
#    plt.title(title, fontsize=20)
#    plt.axis('off')
   
#    # Save the montage to a file
#    plt.savefig('milestone_3/results/' + title, bbox_inches='tight', pad_inches=0)
#    plt.close()  # Close the figure after saving


# create_montage(good_detection_images, (384, 286), 5, "Face Features Detected")

# create_montage(terrible_face_images, (100,100), 2, "Face Featured Detected with False Positives")

# create_montage(non_face_images, (100,100), 6, "Non-Face Image Features Detected")