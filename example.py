import cv2
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

images = []

for i in range(0, 1522):  
    image_path = f'face-database/BioID_{i:04}.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: Could not load image at {image_path}")
    else:
        images.append(image)
        window_size = (100, 100)  # Size of the sliding window
        thresholds = [200000, 200000, 200000, 200000]  # Thresholds for each feature (adjust as needed)

        face = face_classifier.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        text = "FACE"
        cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        plt.figure(figsize=(20,10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        plt.close()
        
