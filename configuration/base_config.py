import os


class BaseConfig:
    PROJECT_ROOT = os.path.abspath('..')
    ROTEM = PROJECT_ROOT + "\\" + "core\\cnn_model_keras.h5"
    ROTEM2 = PROJECT_ROOT + "\\" + "core\\shape_predictor_68_face_landmarks.dat"
    EMOJI_FOLDER = PROJECT_ROOT + "\\" + "emojis\\"
    OUTPUT_MODEL = PROJECT_ROOT + "\\" + "models\\emojis_classifier.h5"
