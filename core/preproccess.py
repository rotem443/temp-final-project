import pickle
import numpy as np
from keras.utils import np_utils
from configuration.config import current_config


class Preprocessors:

    def dataset_splited(self, shape_x, shape_y):

        with open(current_config.TRAIN_IMAGES, "rb") as f:
            train_images = np.array(pickle.load(f))
        with open(current_config.TRAIN_LABELS, "rb") as f:
            train_labels = np.array(pickle.load(f), dtype=np.uint8)

        with open(current_config.TEST_IMAGES, "rb") as f:
            test_images = np.array(pickle.load(f))
        with open(current_config.TEST_LABELS, "rb") as f:
            test_labels = np.array(pickle.load(f), dtype=np.uint8)

        with open(current_config.VAL_IMAGES, "rb") as f:
            val_images = np.array(pickle.load(f))
        with open(current_config.VAL_LABELS, "rb") as f:
            val_labels = np.array(pickle.load(f), dtype=np.uint8)

        train_images = np.reshape(train_images, (train_images.shape[0], shape_x, shape_y, 1))
        test_images = np.reshape(test_images, (test_images.shape[0], shape_x, shape_y, 1))
        val_images = np.reshape(val_images, (val_images.shape[0], shape_x, shape_y, 1))

        train_labels = np_utils.to_categorical(train_labels)
        test_labels = np_utils.to_categorical(test_labels)
        val_labels = np_utils.to_categorical(val_labels)

        result = {}

        result['IMAGES']['TEST'] = test_images
        result['IMAGES']['TRAIN'] = train_images
        result['IMAGES']['VAL'] = val_images

        result['LABELS']['TEST'] = test_labels
        result['LABELS']['TRAIN'] = train_labels
        result['LABELS']['VAL'] = val_labels

        return result
