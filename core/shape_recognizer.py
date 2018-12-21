import cv2
from configuration.config import current_config
from imutils import face_utils
from imutils.face_utils import FaceAligner
import dlib


class ShapeRecognizer(object):

    def __init__(self):
        self.shape_predictor_68 = dlib.shape_predictor(current_config.SHAPE_PREDICTOR_68)
        self.face_aligner = FaceAligner(self.shape_predictor_68, desiredFaceWidth=250)
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        return faces, gray

    def shape_and_align(self, face, img, gray):
        shape_68 = self.shape_predictor_68(img, face)
        shape = face_utils.shape_to_np(shape_68)

        mask = create_mask(shape, img)
        masked = cv2.bitwise_and(gray, mask)

        mask_aligned = self.face_aligner.align(mask, gray, face)
        face_aligned = self.face_aligner.align(masked, gray, face)
        (x0, y0, x1, y1) = get_bounding_rect(mask_aligned)
        face_aligned = face_aligned[y0:y1, x0:x1]
        face_aligned = cv2.resize(face_aligned, (100, 100))

        return face_aligned