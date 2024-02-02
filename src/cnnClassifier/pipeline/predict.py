import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from typing import List, Dict


class PredictionPipeline:
    """
    A pipeline for making predictions on input images using a pre-trained model.

    Args:
        filename (str): The filename of the input image.

    Methods:
        __init__: Initializes the PredictionPipeline object.
        predict: Takes an image and returns the prediction as a list of dictionaries.

    """
    def __init__(self, filename:str) -> None:
        """
        Initializes the PredictionPipeline object.

        Args:
            filename (str): The filename of the input image.
        """
        self.filename = filename

    def predict(self) -> List[dict[str, str]]:
        """
        Takes an image and returns the prediction as a list of dictionaries.

        Returns:
            A list containing a dictionary with the prediction for the input image.
            The dictionary has a single key "image" with the corresponding prediction value.
        """

        # load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Healthy"
            return [{"image": prediction}]
        else:
            prediction = "Coccidiosis"
            return [{"image": prediction}]
