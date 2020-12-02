from os import path, listdir, environ
from sys import argv
import tensorflow as tf
import numpy
from numpy import ndarray
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_restful.inputs import boolean


# Variables
debug = False


# Function for debug logging
def logDebug(text):
    if debug:
        print(str(text))


# ---=HeroAI Interface=--- #
# Fix for finding the dnn implementation
environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Global variable used to store the model that's loaded on the initial run
model = None


def getTrainedModel(file):
    model = tf.keras.models.load_model(file)
    model.summary()
    return model


def newestModelFile():
    base_dir = "models/deep_models/"
    models = [model for model in listdir(base_dir) if path.isdir(base_dir + model)]
    models.sort()
    return base_dir + models[-1]


def getModel():
    global model

    logDebug("Fetching the newest model...")

    if model:
        return model
    else:
        # Get the newest model
        model_file = newestModelFile()
        model = getTrainedModel(model_file)

        return model


def predictToxicity(texts: ndarray):
    global model

    logDebug("Model: " + str(model))

    # Check for model
    if model:
        logDebug("Model found, predicting...")
        logDebug(texts)

        tensor_predictions = model(texts)

        logDebug(tensor_predictions)

        for prediction in tensor_predictions.numpy():
            logDebug("Toxicity prediction: {:2.0f}%".format(100*prediction[0]))
        return tensor_predictions
    else:
        logDebug("No model, getting model...")
        model = getModel()
        predictToxicity(texts)


# Get the newest model on run of the API script
getModel()


# ---=Flask REST API=--- #

# The Flask API Application
app = Flask("HeroAI-API")
api = Api(app)


# The resource giving the toxicity value from the AI
class Toxicity(Resource):
    """
    Resource to get the toxicity of a given text.
    """

    parser = reqparse.RequestParser()
    parser.add_argument("text", type=str, required=True, action="append", help="Please include a classification "
                                                                               "string(s) in the request body. "
                                                                               "- {error_msg}")
    parser.add_argument("raw", type=boolean, help="Whether to get raw prediction values as well.")

    def get(self):
        args = self.parser.parse_args()

        texts = numpy.asarray(args['text'])

        raw = False
        if "raw" in args and args['raw']:
            raw = args['raw']

        tensor_predictions = predictToxicity(texts)
        predictions = [prediction[0] for prediction in tensor_predictions.numpy()]

        # Convert tensor predictions to percentage predictions
        percentage_predictions = [int(round(100*prediction, 0)) for prediction in predictions]

        if raw:
            toxicity_rating = [{"text": texts[i], "toxicity": percentage_predictions[i], "raw_toxicity": predictions[i].astype(float)} for i in range(len(predictions))]
        else:
            toxicity_rating = [{"text": texts[i], "toxicity": percentage_predictions[i]} for i in range(len(predictions))]

        return toxicity_rating, 200


api.add_resource(Toxicity, '/toxicity')


# ---=Python Script Run=--- #


# Function to run the server
def runServer():
    app.run(debug=debug)


if __name__ == "__main__":
    # If an argument is parsed, we use that instead
    if len(argv) > 1:
        argv.pop(0)  # Remove the argument with the script itself

        # Check if it's to run the API server, else just try and predict the given text
        if str(argv[0]).lower() == "runserver":
            runServer()
        else:
            text = " ".join([str(arg) for arg in argv])
            texts = numpy.array([text])

            predictToxicity(texts)
    else:
        runServer()

        # predictToxicity(numpy.asarray(["fuck you karen", "have a nice day"]))
