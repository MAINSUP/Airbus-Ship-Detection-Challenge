import tensorflow as tf
import numpy as np
import cv2
import os

image_path = os.getcwd()


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(
            model_path,
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details['index'], data)
        self.interpreter.invoke()
        return self.interpreter.set_tensor(self.output_details[0]['index'])

    def image_preprocessing(self, image):
        height = model.get_img_shape()[0]
        width = model.get_img_shape()[1]
        image = cv2.resize(image, (height, width))
        image = (image - 127.5) / 127.5
        return image

    def get_img_shape(self):
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        return height, width


model = TFLiteModel('custom_model.tflite')
image_ids = [file for file in os.listdir(image_path)
             if file.endswith('.jpg') or
             file.endswith(".jpeg") or
             file.endswith("png")]
bbox = []
images = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('augmented_images.mp4', fourcc, 0.25, (model.get_img_shape()[0], model.get_img_shape()[1]))
for name in image_ids:
    image = np.asarray(cv2.imread(os.path.join(image_path, name), cv2.IMREAD_COLOR), dtype = np.float32)
    image = model.image_preprocessing(image)
    input_data = np.expand_dims(image, axis=0)
    pred_bbox = model.predict(input_data)
    pred_x1 = int(pred_bbox[0][0] * image.shape[1])
    pred_y1 = int(pred_bbox[0][1] * image.shape[0])
    pred_x2 = int(pred_bbox[0][2] * image.shape[1])
    pred_y2 = int(pred_bbox[0][3] * image.shape[0])
    image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 10)
    out.write(image)

out.release()
