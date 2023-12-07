import cv2  
import numpy as np
from keras.models import load_model  
import tensorflow

class Classifier:

    def __init__(self,Model_path=None,Label_path=None):

        np.set_printoptions(suppress=True)
        self.Model_path = Model_path
        self.Label_path = Label_path
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.model = 0
        if self.Model_path:
            self.model = load_model(self.Model_path,)
        else:
            print("No Model Found")
        if self.Label_path:
            label_file = open(self.Label_path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self,image):

        img_reshaped = cv2.resize(image, (224, 224))
        image_array = np.asarray(img_reshaped)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        return list(prediction[0]), indexVal
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Initialize video capture
    Model_path = "Model\keras_model.h5"
    Label_path = "Model\labels.txt"
    maskClassifier = Classifier(Model_path)

    while True:
        _, img = cap.read()  # Capture frame-by-frame
        # prediction = maskClassifier.getPrediction(img)
        # print(prediction)  # Print prediction result
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Wait for a key press
        
