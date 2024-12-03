import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Haar cascade classifier
face_classifier = cv2.CascadeClassifier(
   cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Lists to store full images with bounding boxes and cropped faces
images_with_boxes = []
faces_detected = []

# Process the first 20 images
for i in range(20):  
   image_path = f'face-database/BioID_{i:04}.pgm'
   image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   # Check if the image was loaded successfully
   if image is None:
       print(f"Warning: Could not load image at {image_path}")
       continue

   # Detect faces
   faces = face_classifier.detectMultiScale(
       image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
   )

   # Draw bounding boxes on a copy of the image
   image_with_boxes = image.copy()
   for (x, y, w, h) in faces:
       cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
       face = image[y:y+h, x:x+w]  # Extract the face region
       resized_face = cv2.resize(face, (100, 100))  # Resize for montage
       faces_detected.append(resized_face)

   if len(faces) > 0:
       images_with_boxes.append(cv2.resize(image_with_boxes, (200, 200)))  # Resize for montage

# Create montages
def create_montage(images, tile_size, grid_cols, title):
   num_images = len(images)
   grid_rows = (num_images + grid_cols - 1) // grid_cols  # Calculate rows
   montage_height = grid_rows * tile_size[1]
   montage_width = grid_cols * tile_size[0]
   montage = np.zeros((montage_height, montage_width), dtype=np.uint8)

   for idx, img in enumerate(images):
       row, col = divmod(idx, grid_cols)
       y_start, x_start = row * tile_size[1], col * tile_size[0]
       montage[y_start:y_start+tile_size[1], x_start:x_start+tile_size[0]] = img

   plt.figure(figsize=(20, 10))
   plt.imshow(montage, cmap='gray')
   plt.title(title, fontsize=20)
   plt.axis('off')
   
   # Save the montage to a file
   plt.savefig('milestone_1/results/' + title, bbox_inches='tight', pad_inches=0)
   plt.close()  # Close the figure after saving

# Display both montages
if images_with_boxes:
   create_montage(images_with_boxes, (200, 200), 5, "Images with Bounding Boxes")
else:
   print("No faces detected for bounding box montage.")

if faces_detected:
   create_montage(faces_detected, (100, 100), 5, "Cropped Face Regions")
else:
   print("No faces detected for cropped face montage.")