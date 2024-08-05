import cv2
import numpy as np

class CannyInference:

    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward_each(self, img):
        numpy_img = img.numpy().astype(np.uint8).transpose(1, 2, 0)
        return cv2.Canny(numpy_img, self.low_threshold, self.high_threshold)[np.newaxis, :, :]

    def __call__(self, images):
        preds = []
        for image in images:
            preds.append(self.forward_each(image))
        return np.stack(preds)