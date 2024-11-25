###############################
#
# This code created by Anan P.
#
##############################

import os
import warnings
import cv2
import mediapipe as mp
import math

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Optional: Suppress any further Abseil warnings/errors from TensorFlow Lite and other libraries
from absl import logging
logging.set_verbosity(logging.ERROR)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to rotate the image around a specific point (midpoint between eyes)
def rotate_image(image, angle, center_point):
    h, w = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image

# Function to calculate the angle between two points
def calculate_angle(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

# Function to resize image to 320px width while maintaining aspect ratio
def resize_image(image, target_width=320):
    h, w = image.shape[:2]
    target_height = int((target_width / w) * h)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

# Function to save an image with an approximate file size of 80KB
def save_image_with_size_control(image, file_path, target_size=80 * 1024):
    quality = 90  # Initial quality setting
    file_size = 0

    # Save the image with compression, and dynamically adjust quality to get ~80KB size
    while file_size < 75 * 1024 or file_size > 85 * 1024:
        # Save the image as a JPEG file
        cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        file_size = os.path.getsize(file_path)

        # Adjust quality based on file size
        if file_size > 85 * 1024:
            quality -= 5  # Reduce quality to reduce file size
        elif file_size < 75 * 1024:
            quality += 5  # Increase quality to increase file size

        # Prevent extreme quality settings
        if quality <= 10 or quality >= 95:
            break

    print(f"Saved image: {file_path} with size: {file_size / 1024:.2f} KB")

# Function to add numbered landmarks to the image
def add_landmarks_to_image(image, face_landmarks):
    h, w, _ = image.shape
    for idx, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Draw landmark
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # Add number near landmark
        #cv2.putText(image, str(idx + 1), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

# Function to crop face area with 30px padding
def crop_face_with_padding(image, face_landmarks, padding=30):
    h, w, _ = image.shape
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Iterate through all landmarks to find the bounding box
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Apply padding, making sure it stays within the image boundaries
    min_x = max(min_x - padding, 0)
    min_y = max(min_y - padding, 0)
    max_x = min(max_x + padding, w)
    max_y = min(max_y + padding, h)

    # Crop the image
    cropped_face = image[min_y:max_y, min_x:max_x]
    return cropped_face

# Reapply landmarks after rotation
def reapply_landmarks_on_rotated_image(rotated_image):
    # Convert the rotated image to RGB before processing
    rgb_rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    # Reprocess the image to find face landmarks
    rotated_result = face_mesh.process(rgb_rotated_image)

    return rotated_result.multi_face_landmarks if rotated_result.multi_face_landmarks else None

# Read the image (ensure the image path is correct)
image_path = r"./images/test.png"
image = cv2.imread(image_path)

# Get the base filename (without extension) for saving outputs
base_name = os.path.basename(image_path)
file_name, file_ext = os.path.splitext(base_name)

# Check if the image was successfully loaded
if image is None:
    print("Error: Image not found or path is incorrect.")
else:
    # Convert the BGR image to RGB before processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    result = face_mesh.process(rgb_image)

    # Check if faces are detected
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get the coordinates of landmarks 33 (left eye center) and 263 (right eye center)
            h, w, _ = image.shape  # Get height and width of the image
            point_33 = face_landmarks.landmark[33]  # Left eye center
            point_263 = face_landmarks.landmark[263]  # Right eye center

            # Convert the normalized coordinates to pixel coordinates
            x_33, y_33 = int(point_33.x * w), int(point_33.y * h)
            x_263, y_263 = int(point_263.x * w), int(point_263.y * h)

            # Calculate the midpoint between the two points
            midpoint_x = (x_33 + x_263) // 2
            midpoint_y = (y_33 + y_263) // 2
            midpoint = (midpoint_x, midpoint_y)

            # Calculate the angle between the two points
            angle = calculate_angle((x_33, y_33), (x_263, y_263))  # Left to right eye
            print(f"Calculated angle between eye centers: {angle} degrees")

            # Rotate the image to make the eyes parallel (rotate around the midpoint)
            rotated_image = rotate_image(image, angle, midpoint)

            # Reapply landmarks on the rotated image
            new_face_landmarks = reapply_landmarks_on_rotated_image(rotated_image)

            if new_face_landmarks:
                for new_face_landmark in new_face_landmarks:
                    # Crop the face with 30px padding
                    cropped_face = crop_face_with_padding(rotated_image, new_face_landmark)

                    # Save the cropped face with 30px padding as JPEG (ensure the correct file size)
                    cropped_face_output_path = os.path.join(os.path.dirname(image_path), f"{file_name}_face.jpg")
                    save_image_with_size_control(cropped_face, cropped_face_output_path)

                    # Resize the rotated image to 320px width while maintaining aspect ratio
                    resized_image = resize_image(rotated_image)

                    # Save the rotated and resized image (without landmarks) as JPEG
                    output_image_rotated_path = os.path.join(os.path.dirname(image_path), f"{file_name}_eyes_aligned.jpg")
                    save_image_with_size_control(resized_image, output_image_rotated_path)

                    # Add numbered landmarks to the rotated image
                    rotated_image_with_landmarks = rotated_image.copy()
                    add_landmarks_to_image(rotated_image_with_landmarks, new_face_landmark)

                    # Resize the image with landmarks to 320px width while maintaining aspect ratio
                    resized_image_with_landmarks = resize_image(rotated_image_with_landmarks)

                    # Save the rotated image with landmarks (numbered) as JPEG
                    output_image_with_landmarks_path = os.path.join(os.path.dirname(image_path), f"{file_name}_eyes_aligned_with_landmarks.jpg")
                    save_image_with_size_control(resized_image_with_landmarks, output_image_with_landmarks_path)
            else:
                print("No landmarks detected after rotation.")

    else:
        print("No faces detected.")

    # Cleanup
    cv2.destroyAllWindows()
