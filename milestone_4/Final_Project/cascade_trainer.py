import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import random
import os
from tqdm import tqdm  # For progress bars

# Basic Haar feature calculations - these form the building blocks of face detection
def compute_integral_image(img):
    """Convert image to integral image for faster feature computation"""
    return cv2.integral(img)

def haar_feature_two_horizontal(integral_img, x, y, width, height):
    """Compute two-rectangle horizontal feature - looks for horizontal edges"""
    left_sum = integral_img[y + height, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height, x] + integral_img[y, x]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + width // 2] + integral_img[y, x + width // 2]
    return left_sum - right_sum

def haar_feature_two_vertical(integral_img, x, y, width, height):
    """Compute two-rectangle vertical feature - looks for vertical edges"""
    top_sum = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
              - integral_img[y + height // 2, x] + integral_img[y, x]
    bottom_sum = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                 - integral_img[y + height, x] + integral_img[y + height // 2, x]
    return top_sum - bottom_sum

def haar_feature_three_horizontal(integral_img, x, y, width, height):
    """Compute three-rectangle horizontal feature - looks for centered horizontal patterns"""
    left_sum = integral_img[y + height, x + width // 3] - integral_img[y, x + width // 3] \
               - integral_img[y + height, x] + integral_img[y, x]
    middle_sum = integral_img[y + height, x + 2 * width // 3] - integral_img[y, x + 2 * width // 3] \
                 - integral_img[y + height, x + width // 3] + integral_img[y, x + width // 3]
    right_sum = integral_img[y + height, x + width] - integral_img[y, x + width] \
                - integral_img[y + height, x + 2 * width // 3] + integral_img[y, x + width // 3]
    return left_sum - 2 * middle_sum + right_sum

def haar_feature_four_rectangle(integral_img, x, y, width, height):
    """Compute four-rectangle feature - looks for diagonal patterns"""
    top_left = integral_img[y + height // 2, x + width // 2] - integral_img[y, x + width // 2] \
               - integral_img[y + height // 2, x] + integral_img[y, x]
    top_right = integral_img[y + height // 2, x + width] - integral_img[y, x + width] \
                - integral_img[y + height // 2, x + width // 2] + integral_img[y, x + width // 2]
    bottom_left = integral_img[y + height, x + width // 2] - integral_img[y + height // 2, x + width // 2] \
                  - integral_img[y + height, x] + integral_img[y + height // 2, x]
    bottom_right = integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] \
                   - integral_img[y + height, x + width // 2] + integral_img[y + height // 2, x + width // 2]
    return top_left + bottom_right - top_right - bottom_left

# Trainer Class
class HaarCascadeTrainer:
    def __init__(self, positive_samples, negative_samples, num_stages=5, 
                 min_detection_rate=0.9, max_false_positive_rate=0.4):
        # Store parameters and initialize samples
        self.target_size = (130, 150)  # Fixed size for face detection
        print(f"Resizing positive samples to {self.target_size}...")
        self.positive_samples = [cv2.resize(img, self.target_size) for img in tqdm(positive_samples)]
        self.negative_samples = negative_samples
        self.num_stages = num_stages
        self.min_detection_rate = min_detection_rate
        self.max_false_positive_rate = max_false_positive_rate
        self.stages = []
    
    def compute_features(self, integral_image):
        """Improved feature computation with more variation"""
        features = []
        h, w = integral_image.shape
        window_size = self.target_size
        
        # Use smaller step size for more features
        step = 4  # Changed from 8 to 4
        
        for y in range(0, h - window_size[1], step):
            for x in range(0, w - window_size[0], step):
                # Add variation in feature sizes
                for size_scale in [1.0, 0.75, 0.5]:
                    feat_w = int(window_size[0] * size_scale)
                    feat_h = int(window_size[1] * size_scale)
                    if x + feat_w <= w and y + feat_h <= h:
                        features.extend([
                            haar_feature_two_horizontal(integral_image, x, y, feat_w, feat_h),
                            haar_feature_two_vertical(integral_image, x, y, feat_w, feat_h),
                            haar_feature_three_horizontal(integral_image, x, y, feat_w, feat_h),
                            haar_feature_four_rectangle(integral_image, x, y, feat_w, feat_h)
                        ])
        return features

    def process_training_samples(self):
        """Process all training samples to compute features"""
        print("Processing positive samples...")
        pos_features = []
        # Adding progress bar
        for img in tqdm(self.positive_samples):
            integral = compute_integral_image(img)
            features = self.compute_features(integral)
            pos_features.append(features)

        print("Processing negative samples...")
        neg_features = []
        for img in tqdm(self.negative_samples):
            # Extract multiple windows from each negative image
            h, w = img.shape
            for _ in range(5):  # Get 5 random windows from each negative
                if h > self.target_size[1] and w > self.target_size[0]:
                    y = random.randint(0, h - self.target_size[1])
                    x = random.randint(0, w - self.target_size[0])
                    window = img[y:y+self.target_size[1], x:x+self.target_size[0]]
                    integral = compute_integral_image(window)
                    features = self.compute_features(integral)
                    neg_features.append(features)

        return np.array(pos_features), np.array(neg_features)

    def train_weak_classifier(self, features, labels, weights):
        """
        Train a weak classifier by finding the best feature and threshold.
        Uses weighted error to focus on harder examples.
        
        Args:
            features: Array of computed Haar features
            labels: Binary labels (1 for faces, -1 for non-faces)
            weights: Sample weights updated by AdaBoost
            
        Returns:
            Tuple of (feature_index, threshold, polarity, alpha) for best weak classifier
        """
        best_error = float('inf')
        best_feature_idx = None
        best_threshold = None
        best_polarity = None
        
        # Sample a subset of features for efficiency
        feature_indices = random.sample(range(features.shape[1]), 
                                    min(100, features.shape[1]))
        
        for feature_idx in feature_indices:
            feature_values = features[:, feature_idx]
            
            # Use more conservative threshold selection
            thresholds = np.percentile(feature_values, 
                                    [30, 40, 50, 60, 70])  # More centered thresholds
            
            for threshold in thresholds:
                for polarity in [-1, 1]:
                    predictions = np.ones(len(labels))
                    predictions[polarity * feature_values < polarity * threshold] = -1
                    
                    # Compute weighted error with stronger penalty for false negatives
                    false_negatives = (predictions == -1) & (labels == 1)
                    false_positives = (predictions == 1) & (labels == -1)
                    
                    # Balance errors differently
                    weighted_error = (
                        np.sum(weights[false_negatives]) * 4.0 +  # Heavily penalize missing faces
                        np.sum(weights[false_positives])          # Standard penalty for false positives
                    )
                    
                    if weighted_error < best_error:
                        best_error = weighted_error  
                        best_feature_idx = feature_idx
                        best_threshold = threshold
                        best_polarity = polarity
        
        # Compute a more conservative alpha
        alpha = 0.3 * np.log((1 - best_error + 1e-10) / (best_error + 1e-10))
        return (best_feature_idx, best_threshold, best_polarity, alpha)

    def train(self):
        """
        Train the complete cascade classifier using AdaBoost for each stage.
        
        This implementation uses a multi-stage approach where each stage becomes 
        progressively more selective in detecting faces. The training process:
        1. Processes all samples to compute Haar features
        2. Trains multiple stages of weak classifiers
        3. Maintains detection rate while reducing false positives
        4. Uses adaptive weighting to focus on hard examples
        
        The training stops when either:
        - All stages are trained successfully
        - Detection rate drops too low (early stopping)
        - Target false positive rate is achieved
        
        Training parameters are controlled by:
        - self.min_detection_rate (minimum acceptable detection rate)
        - self.max_false_positive_rate (maximum acceptable false positive rate)
        - self.num_stages (number of cascade stages to train)
        """
        print("Starting cascade training...")
        
        # First, compute all features for efficiency
        # This prevents recomputing features during training
        pos_features, neg_features = self.process_training_samples()
        
        # Initialize tracking array for samples that pass each stage
        # Using boolean type to avoid type conversion issues
        current_predictions = np.ones(len(pos_features) + len(neg_features), dtype=bool)
        
        # Keep track of cumulative performance
        cumulative_dr = 1.0  # Detection Rate
        cumulative_fpr = 1.0  # False Positive Rate
        
        # Train each stage of the cascade
        for stage in range(self.num_stages):
            print(f"\nTraining stage {stage + 1}/{self.num_stages}")
            
            # Get samples that passed all previous stages
            valid_samples = current_predictions
            
            # Prepare features and labels for current stage
            stage_features = np.vstack([pos_features, neg_features])[valid_samples]
            stage_labels = np.hstack([
                np.ones(len(pos_features)),      # Positive samples (faces)
                -np.ones(len(neg_features))      # Negative samples (non-faces)
            ])[valid_samples]
            
            # Initialize weights uniformly
            weights = np.ones(len(stage_labels)) / len(stage_labels)
            
            # Storage for weak classifiers in this stage
            stage_classifiers = []
            
            # Track best performance for this stage
            best_stage_dr = 0.0
            best_stage_fpr = 1.0
            
            # Train weak classifiers for this stage
            for i in range(10):  # Train 10 weak classifiers per stage
                # Train a single weak classifier
                classifier = self.train_weak_classifier(stage_features, stage_labels, weights)
                stage_classifiers.append(classifier)
                
                # Get predictions using all classifiers so far
                predictions = self.apply_stage_to_features(stage_features, stage_classifiers)
                
                # Update sample weights using AdaBoost formula
                # Increase weights of misclassified samples
                weights *= np.exp(-classifier[3] * stage_labels * predictions)
                weights /= np.sum(weights)  # Normalize weights
                
                # Calculate performance metrics
                pos_idx = stage_labels == 1
                stage_dr = np.mean(predictions[pos_idx] == 1)    # Detection Rate
                stage_fpr = np.mean(predictions[~pos_idx] == 1)  # False Positive Rate
                
                print(f"Weak classifier {i+1}: FPR = {stage_fpr:.3f}, DR = {stage_dr:.3f}")
                
                # Track best performance
                if stage_dr > best_stage_dr and stage_fpr < best_stage_fpr:
                    best_stage_dr = stage_dr
                    best_stage_fpr = stage_fpr
                
                # Check if stage met the target rates
                if stage_fpr < self.max_false_positive_rate and stage_dr > self.min_detection_rate:
                    print("Target rates achieved - stopping early for this stage")
                    break
                
                # Early stopping if detection rate drops too low
                if stage_dr < 0.5:  # Stop if we're detecting less than 50% of faces
                    print("Warning: Detection rate too low - rolling back last classifier")
                    stage_classifiers.pop()  # Remove last classifier
                    break
            
            # Add this stage's classifiers to our cascade
            self.stages.append(stage_classifiers)
            
            # Update predictions for all samples
            stage_result = self.apply_stage_to_features(
                np.vstack([pos_features, neg_features])[current_predictions], 
                stage_classifiers
            )
            
            # Update which samples pass this stage
            current_predictions[current_predictions] &= stage_result
            
            # Calculate cumulative performance
            pos_remaining = np.sum(current_predictions[:len(pos_features)])
            neg_remaining = np.sum(current_predictions[len(pos_features):])
            
            cumulative_dr = pos_remaining / len(pos_features)
            cumulative_fpr = neg_remaining / len(neg_features)
            
            # Print stage summary
            print(f"\nStage {stage + 1} Summary:")
            print(f"Positive samples remaining: {pos_remaining}/{len(pos_features)}")
            print(f"Negative samples remaining: {neg_remaining}/{len(neg_features)}")
            print(f"Current detection rate: {cumulative_dr:.3f}")
            print(f"Current false positive rate: {cumulative_fpr:.3f}")
            
            # Check if we should stop training
            if cumulative_dr < 0.5:  # If we've lost too many positive samples
                print("\nWarning: Too many positive samples eliminated - stopping training")
                break
            
            if cumulative_fpr < self.max_false_positive_rate:
                print("\nReached target false positive rate - stopping training")
                break
        
        # Training complete - print final summary
        print("\nCascade training completed!")
        print(f"Final cascade has {len(self.stages)} stages")
        print(f"Total weak classifiers: {sum(len(stage) for stage in self.stages)}")
        print(f"Final detection rate: {cumulative_dr:.3f}")
        print(f"Final false positive rate: {cumulative_fpr:.3f}")
        
    def apply_stage(self, features, stage_classifiers):
        """Apply a single stage of the cascade to features"""
        scores = np.zeros(len(features))
        for feature_idx, threshold, polarity, alpha in stage_classifiers:
            predictions = np.ones(len(features))
            predictions[polarity * features[:, feature_idx] < polarity * threshold] = -1
            scores += alpha * predictions
        return scores >= -0.8
    
    def apply_stage_to_features(self, features, stage_classifiers):
        """
        Apply stage classifiers to a set of features.
        This is similar to apply_stage but works directly with feature arrays.
        
        Args:
            features: Array of computed Haar features
            stage_classifiers: List of weak classifiers for this stage
        
        Returns:
            Binary predictions (1 for face, -1 for non-face)
        """
        scores = np.zeros(len(features))
        for feature_idx, threshold, polarity, alpha in stage_classifiers:
            predictions = np.ones(len(features))
            predictions[polarity * features[:, feature_idx] < polarity * threshold] = -1
            scores += alpha * predictions
        
        # Add a bias term to make the overall stage more lenient
        return scores >= -0.8  # Changed from 0 to make it more lenient

def detect_faces(image, cascade_stages, window_size=(130, 150), scale_factor=1.1, 
                min_neighbors=3, step_ratio=0.05):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    detections = []
    
    scale = 1.0
    while scale * window_size[0] < gray.shape[1] and scale * window_size[1] < gray.shape[0]:
        scaled = cv2.resize(gray, None, fx=1/scale, fy=1/scale)
        integral_img = compute_integral_image(scaled)
        
        step = int(window_size[0] * step_ratio)
        for y in range(0, scaled.shape[0] - window_size[1], step):
            for x in range(0, scaled.shape[1] - window_size[0], step):
                passed = True
                for stage in cascade_stages:
                    stage_sum = 0
                    for feature_idx, threshold, polarity, alpha in stage:
                        # Use all feature types based on feature index
                        feature_type = feature_idx % 4  # Cycle through feature types
                        if feature_type == 0:
                            feature_value = haar_feature_two_horizontal(integral_img, x, y, window_size[0], window_size[1])
                        elif feature_type == 1:
                            feature_value = haar_feature_two_vertical(integral_img, x, y, window_size[0], window_size[1])
                        elif feature_type == 2:
                            feature_value = haar_feature_three_horizontal(integral_img, x, y, window_size[0], window_size[1])
                        else:
                            feature_value = haar_feature_four_rectangle(integral_img, x, y, window_size[0], window_size[1])
                            
                        prediction = 1 if polarity * feature_value >= polarity * threshold else -1
                        stage_sum += alpha * prediction
                    
                    if stage_sum < -0.8:  # Use same threshold as training
                        passed = False
                        break
                
                if passed:
                    detections.append((
                        int(x * scale), 
                        int(y * scale),
                        int(window_size[0] * scale),
                        int(window_size[1] * scale)
                    ))
        
        scale *= scale_factor
    
    return non_maximum_suppression(detections, min_neighbors)

# The idea of non_maximum_suppression was introduced to us by Peter and Nicholas during their Project presentation
def non_maximum_suppression(boxes, min_neighbors):
    """Remove overlapping detections, keeping only the strongest ones"""
    if not boxes:
        return []

    # Convert to numpy array for easier computation
    boxes = np.array(boxes)
    pick = []
    
    # Compute coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute the area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by bottom-right y-coordinate
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        # Pick the last box
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete all indexes we're done with
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > 0.3)[0])))
    
    return boxes[pick].tolist()

def test_detector(cascade_model_path, test_image_dir, output_dir):
    """
    Test the face detector on a directory of images and save the results.
    
    Args:
        cascade_model_path: Path to the saved cascade model
        test_image_dir: Directory containing test images
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained cascade model
    with open(cascade_model_path, 'rb') as f:
        cascade_stages = pickle.load(f)
    
    # Process each image in the test directory
    for filename in os.listdir(test_image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(test_image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {filename}")
                continue
            
            # Detect faces
            faces = detect_faces(img, cascade_stages)
            
            # Draw rectangles around detected faces
            result_img = img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save the result
            output_path = os.path.join(output_dir, f'detected_{filename}')
            cv2.imwrite(output_path, result_img)
            print(f"Processed {filename}: Found {len(faces)} faces")


def main():
    """Main function synthesizes all functions"""
    
    # Initialization For Training:
    # Load positive samples (faces)
    positive_samples = []
    for i in range(1800):  # Load 1800 positive samples
        img = cv2.imread(f'data/faces/img_{i:04d}.jpg', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            positive_samples.append(img)
    
    # Load negative samples (non-faces)
    negative_samples = []
    for i in range(7500):  # Load 7500 negative samples
        img = cv2.imread(f'data/non_faces/img_{i:04d}.jpg', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            negative_samples.append(img)
    
    # Choose operation mode
    print("\nFace Detection System")
    print("1. Train new model and save cascade stages")
    print("2. Load existing model and detect faces from test images")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        print("\nStarting training phase...")
        # Train cascade classifier
        trainer = HaarCascadeTrainer(positive_samples, negative_samples)
        trainer.train()
        
        # Save the trained model with pickle
        with open('cascade_model.pkl', 'wb') as f:
            pickle.dump(trainer.stages, f)
        print("Model saved successfully!")
        
        # Testing!
    if choice == '2':
        print("\nStarting testing phase...")
        test_detector(
            cascade_model_path='cascade_model.pkl',
            test_image_dir='data/test_images',
            output_dir='data/results'
        )
        print("Testing Completed Sucessfully!")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training/testing phase: {e}")
    
    
    