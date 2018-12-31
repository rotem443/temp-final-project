import os
import cv2
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from configuration.config import current_config
from keras.layers.normalization import BatchNormalization
from keras import backend as keras_backend
from keras.models import load_model
import numpy as np

keras_backend.set_image_dim_ordering('tf')


class ModelHandler:

    def __init__(self, pre_processor, shape_recognizer):
        self.pre_processor = pre_processor
        self.shape_recognizer = shape_recognizer

    def _get_image_size(self):
        img = cv2.imread('dataset/0/1.jpg', 0)
        return img.shape

    def _get_num_of_classes(self):
        return len(os.listdir('dataset/'))

    def _process_image(self, img, input_shape_x, input_shape_y):
        img = cv2.resize(img, (input_shape_x, input_shape_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (1, input_shape_x, input_shape_y, 1))

        return img

    def generate_cnn_model(self, input_shape_x: int, input_shape_y: int):
        num_of_classes = self._get_num_of_classes()

        model = Sequential()

        model.add(Conv2D(32, (5, 5), input_shape=(input_shape_x, input_shape_y, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Dense(num_of_classes, activation='softmax'))

        sgd = optimizers.SGD(lr=1e-2)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model

    def train(self, evaluate: bool):

        input_shape_x, input_shape_y = self._get_image_size()

        processed_data = self.pre_processor.get_dataset_splited(input_shape_x, input_shape_y)

        model = self.generate_cnn_model(input_shape_x, input_shape_y)

        model.fit(processed_data['IMAGES']['TRAIN'],
                  processed_data['LABELS']['TRAIN'],
                  validation_data=(processed_data['IMAGES']['TEST'], processed_data['LABELS']['TEST']),
                  epochs=15,
                  batch_size=100)

        model = load_model(current_config.OUTPUT_MODEL)

        if evaluate:
            scores = model.evaluate(processed_data['IMAGES']['VAL'],
                                    processed_data['LABELS']['VAL'],
                                    verbose=1)

            print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    def predict(self, model, image):

        input_shape_x, input_shape_y = self._get_image_size()

        processed = self._process_image(image, input_shape_x, input_shape_y)
        predication = model.predict(processed)
        predication_probability = predication[0]
        predication_class = list(predication_probability).index(max(predication_probability))

        return max(predication_probability), predication_class
