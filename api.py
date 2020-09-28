import json
from os import path, listdir
import tensorflow as tf

tokenizer = None
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


def predictToxicity(text, tokenized=False):
    global tokenizer
    global model

    print("Tokenizer: ")
    print(tokenizer)
    print("Model: ")
    print(model)
    print("Tokenized: ")
    print(tokenized)

    # Check for tokenizer first
    if not tokenizer:
        print("No tokenizer, getting tokenizer...")
        tokenizer = getTokenizer()

    # Tokenize if not tokenized
    if tokenizer and not tokenized:
        print("Tokenizer found, not tokenized, tokenizing...")
        text_sequence = tokenizer.texts_to_sequences([text])
        print(text_sequence)
        decoded_text = tokenizer.sequences_to_texts(text_sequence)
        print(decoded_text)

        text_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=5000)

        tokenized = True
        text = text_sequence_padded

    # Check for model
    if model:
        print("Model found, predicting...")
        prediction = model(text)
        print("Toxicity prediction: {:2.0f}%".format(100*prediction.numpy()[0][0]))
        return prediction
    else:
        print("No model, getting model...")
        model_file = newestModelFile()
        model = getTrainedModel(model_file)
        predictToxicity(text, tokenized)


if __name__ == "__main__":
    text = "Fuck you Karen!"
    prediction = predictToxicity(text)