import cv2
import os
from core.shape_recognizer import ShapeRecognizer


def create_dataset_from_faces(faces_dataset_path: str,
                              final_dataset_path: str,
                              shape_recognizer: ShapeRecognizer):

    folders = os.listdir(faces_dataset_path)

    for folder in folders:
        folder = folder + '/'

        for image_path in os.listdir(faces_dataset_path + folder):
            path = faces_dataset_path + folder + image_path
            img = cv2.imread(path)

            faces, gray = shape_recognizer.detect_faces(img)

            if len(faces) > 0:
                for face in faces:

                    face_aligned = shape_recognizer.shape_and_align(face, img, gray)

                    cv2.imwrite(final_dataset_path + folder + image_path, face_aligned)
