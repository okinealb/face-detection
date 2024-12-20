import os
from PIL import Image

# Set the folder containing the images
source_directory = "negatives"
# Set the centralized folder where cropped images will be saved
destination_directory = "negatives_final"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Initialize a counter for renaming images
image_counter = 0

# Iterate through each file in the source directory
for image_file in os.listdir(source_directory):
    source_image_path = os.path.join(source_directory, image_file)

    # Check if it's an image file (adjust extensions as needed)
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        try:
            with Image.open(source_image_path) as img:
                # Check the size of the image
                width, height = img.size
                if width >= 250 and height >= 250:
                    # Crop the image to 250x250 pixels
                    cropped_img = img.crop((0, 0, 250, 250))
                else:
                    # Resize the image to 250x250 pixels
                    cropped_img = img.resize((250, 250))

                # Save the cropped or resized image to the destination folder with a new name
                new_image_name = f"img_{image_counter:04}.jpg"  # Adjust extension if needed
                destination_image_path = os.path.join(destination_directory, new_image_name)
                cropped_img.save(destination_image_path)

                print(f"Processed and saved {new_image_name}")

                # Increment the counter
                image_counter += 1
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

print("All images have been processed and saved.")
