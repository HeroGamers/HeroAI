from os import path, listdir, environ
import tensorflow as tf
import numpy

# Fix for finding the dnn implementation
environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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


def predictToxicity(text):
    global model

    print("Model: " + str(model))

    # Check for model
    if model:
        print("Model found, predicting...")
        print(text)
        prediction = model(numpy.array([(text)]))
        print("Toxicity prediction: {:2.0f}%".format(100*prediction.numpy()[0][0]))
        return prediction
    else:
        print("No model, getting model...")
        model_file = newestModelFile()
        model = getTrainedModel(model_file)
        predictToxicity(text)


if __name__ == "__main__":
    text = "I hope you will have a nice"
    prediction = predictToxicity(text)
