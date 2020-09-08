import json
import pandas as pd
from os import path
import tensorflow as tf


def readDatset(path):
    try:
        dataset = pd.read_csv(path, index_col=0)
        if not dataset.empty:
            # Dataset preview
            head = dataset.head()
            print(head)

            return dataset, True
        else:
            print("Dataset is empty!")
    except Exception as e:
        print("Error reading the dataset! - " + str(e))
    return None, False


def slicesFromPanda(file):
    dataset, found = readDatset(file)
    if found:
        dataset_slices = tf.data.Dataset.from_tensor_slices(dict(dataset))
        for feature_batch in dataset_slices.take(1):
            for key, value in feature_batch.items():
                print("  {!r:20s}: {}".format(key, value))
        return dataset_slices


def datasetFromTensor(file, batch_size, buffer_size):
    dataset = tf.data.experimental.make_csv_dataset(file, label_name="toxic", select_columns=["comment_text", "toxic"],
                                                    shuffle_buffer_size=buffer_size, batch_size=batch_size)

    # for feature_batch, label_batch in dataset.take(1):
    #     print("Toxic: {}".format(label_batch))
    #     for key, value in feature_batch.items():
    #         print("  {!r:20s}: {}".format(key, value))
    return dataset


def tokenizeDataset(dataset, length):
    # We use the tokenizer to take all the unique words, and make a dictionary where we assign a value to each word, so
    # that it can be used in the model
    tokenizer_path = "tokenizer.json"
    tokenizer_json = None
    if path.isfile(tokenizer_path):
        with open(tokenizer_path, "r") as token_file:
            tokenizer_json = json.load(token_file)

    if tokenizer_json:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # Get text from dataset, put into list
    print("Putting stuff into a list")
    text_list = []
    i = 1
    for text_tensor, _ in dataset:
        if i == 10:
            break
        print(text_tensor.numpy())
        text_list.append(text_tensor.numpy())
        i += 1

    # Update the token with the dataset values
    print(dataset)

    print("Fitting tokenizer onto the dataset...")
    tokenizer.fit_on_texts(text_list)
    print("Done fitting tokenizer onto dataset!")

    # Tokenize the dataset
    dataset_sequence = tokenizer.texts_to_sequences(text_list)

    return None


def getDataset(dataset, batch_size=64, buffer_size=10000):
    base_path = "datasets/"

    load_dir = None
    train_file = None
    test_content_file = None
    test_labels_file = None
    if dataset == "jigsaw-1":
        load_dir = base_path + "jigsaw-toxic-comment-classification-challenge/"
        train_file = load_dir + "train.csv"
        test_content_file = load_dir + "test.csv"
        test_labels_file = load_dir + "test_labels.csv"

    if load_dir:
        if train_file:
            # slice = slicesFromPanda(train_file)

            dataset = datasetFromTensor(train_file, batch_size, buffer_size)
            tokenized_dataset = tokenizeDataset(dataset, 1500)

    else:
        print("Not a valid dataset!")
    return None


if __name__ == "__main__":
    getDataset("jigsaw-1")
    # readDatset("datasets/jigsaw-toxic-comment-classification-challenge/train.csv")
