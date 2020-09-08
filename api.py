import json
from os import path
import tensorflow as tf


def getTrainedModel(file):
    return None


def getTokenizer():
    tokenizer_path = "tokenizer.json"
    tokenizer_json = None
    if path.isfile(tokenizer_path):
        with open(tokenizer_path, "r") as token_file:
            tokenizer_json = json.load(token_file)

    if tokenizer_json:
        return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    else:
        print("No tokenizer found, please provide one...")
        return None


def predictToxicity(text):
    tokenizer = getTokenizer()
    if tokenizer:
        tokenizer.fit_on_texts([text])
        text_sequence = tokenizer.texts_to_sequences([text])

        model = getTrainedModel("aa")

        if model:
            return model.predict(text_sequence)
