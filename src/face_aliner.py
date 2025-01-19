#############################
# This code create my Anan p.
# hs1gab@gmail.com 
# 19/1/2025
# Example:
#     def alinment_face(self, frame ):
#       aliner = EyeHorizonAligner() 
#       ali_frame = aliner.process_frame(frame)
#       rotage_frame = self.find_largest_face(ali_frame)
#       return rotage_frame
#############################
import os
import cv2
import mediapipe as mp
import math
import warnings


class EyeHorizonAligner:
    def __init__(self):
        """Initialize the EyeHorizonAligner with MediaPipe FaceMesh."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
        from absl import logging
        logging.set_verbosity(logging.ERROR)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh()

    def calculate_angle(self, p1, p2):
        """Calculate the angle between two points."""
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = math.degrees(math.atan2(delta_y, delta_x))
        return angle

    def rotate_image(self, image, angle, center_point):
        """Rotate the image around a specific point."""
        h, w = image.shape[:2]
        rot_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated_image

    def reapply_landmarks(self, rotated_image):
        """Reapply landmarks to a rotated image."""
        rgb_rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        rotated_result = self.face_mesh.process(rgb_rotated_image)
        return rotated_result.multi_face_landmarks if rotated_result.multi_face_landmarks else None

    def process_frame(self, frame):
        """Process a single frame to align the eyes horizontally."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            print("No faces detected.")
            return frame

        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye = face_landmarks.landmark[33]  # Left eye center
            right_eye = face_landmarks.landmark[263]  # Right eye center

            x_left, y_left = int(left_eye.x * w), int(left_eye.y * h)
            x_right, y_right = int(right_eye.x * w), int(right_eye.y * h)

            midpoint = ((x_left + x_right) // 2, (y_left + y_right) // 2)
            angle = self.calculate_angle((x_left, y_left), (x_right, y_right))

            print(f"Calculated angle: {angle} degrees")

            rotated_frame = self.rotate_image(frame, angle, midpoint)
            new_landmarks = self.reapply_landmarks(rotated_frame)

            if new_landmarks:
                print("Landmarks reapplied successfully.")
                return rotated_frame

        print("Failed to reapply landmarks after rotation.")
        return frame
