import json
import pandas as pd
from pandas import DataFrame
from os import path
import tensorflow as tf


# Function to preprocess our datasets
def preprocessPandasDataFrame(df: DataFrame, dataset_name):
    if dataset_name == "jigsaw-1":
        df['text'].replace('''([!"#$%&()*+,\-./':;<=>?@[\]^_`{|}~])''', '', regex=True, inplace=True)
        df['text'].replace('([\n\t])', ' ', regex=True, inplace=True)
    return df


def readDataset(path, dataset_name):
    try:
        dataset = pd.read_csv(path, index_col=0, encoding='utf-8')
        if not dataset.empty:
            # Dataset preview
            head = dataset.head()
            print(head)

            # Rename columns
            if dataset_name == "jigsaw-1":
                dataset.rename(columns={"comment_text": "text"}, inplace=True)

            print("Keeping text and label columns...")
            dataset = dataset[["text", "toxic"]]

            # Preprocess the dataset
            print("Preprocessing dataset...")
            dataset = preprocessPandasDataFrame(dataset, dataset_name)
            print("Done preprocessing the dataset!")

            return dataset, True
        else:
            print("Dataset is empty!")
    except Exception as e:
        print("Error reading the dataset! - " + str(e))
    return None, False


def slicesFromPanda(file, dataset_name):
    dataset, found = readDataset(file, dataset_name)
    if found:
        dataset_slices = tf.data.Dataset.from_tensor_slices((dataset['text'], dataset['toxic']))

        print(dataset_slices)

        for text, label in dataset_slices.take(4):
            print("  'text'     : ", text.numpy())
            print("  'toxic'    : ", label.numpy())

        return dataset_slices


def datasetFromTensor(file, batch_size, buffer_size):
    dataset = tf.data.experimental.make_csv_dataset(file, label_name="toxic", select_columns=["comment_text", "toxic"],
                                                    shuffle_buffer_size=buffer_size, batch_size=batch_size)

    # for feature_batch, label_batch in dataset.take(1):
    #     print("Toxic: {}".format(label_batch))
    #     for key, value in feature_batch.items():
    #         print("  {!r:20s}: {}".format(key, value))
    return dataset


# Not used anymore - we tokenize in the model
def tokenizeDataset(dataset):
    # We use the tokenizer to take all the unique words, and make a dictionary where we assign a value to each word, so
    # that it can be used in the model
    tokenizer_path = "tokenizer.json"
    tokenizer_json = None
    if path.isfile(tokenizer_path):
        with open(tokenizer_path, "r", encoding='utf-8') as token_file:
            tokenizer_json = json.load(token_file)

    if tokenizer_json:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # Get text from dataset, put into list
    print("Putting stuff into a list")
    text_list = []
    toxicity_sequence = []
    for feature_batch in dataset:
        isText = True
        for _, value in feature_batch.items():
            if isText:
                text = str(value.numpy(), 'utf-8')  # It is currently bytes, convert to string
                text_list.append(text)
                isText = False
            else:
                toxicity = value.numpy()
                toxicity_sequence.append(toxicity)
                isText = True

    # Get longest text in the list...
    max_length = len(max(text_list, key=len))
    print("Done listing everything!")

    # Update the token with the dataset values
    print("Fitting tokenizer onto the dataset...")
    tokenizer.fit_on_texts(text_list)
    print("Done fitting tokenizer onto dataset!")

    # Save the tokenizer
    print("Saving the tokenizer...")
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, "w", encoding='utf-8') as token_file:
        json.dump(tokenizer_json, token_file, ensure_ascii=False)
    print("Done saving the tokenizer!")

    # Tokenize the dataset
    print("Tokenizing dataset...")
    text_sequence = tokenizer.texts_to_sequences(text_list)
    print("Done tokenizing dataset!")

    # Pad the sequence
    print("Padding the dataset with zeros...")
    text_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=max_length)
    print("Done padding the dataset!")

    # print(text_sequence_padded)
    #
    # print("Test")
    # print(text_sequence_padded[0])
    #
    # text = tokenizer.sequences_to_texts([text_sequence_padded[0]])
    #
    # print(text)

    print("Converting into a new dataset...")
    tokenized_dataset = tf.data.Dataset.from_tensor_slices((text_sequence_padded, toxicity_sequence))
    print("Converted!")

    return tokenized_dataset, len(tokenizer.word_index), max_length


def getDataset(dataset):
    base_path = "datasets/"

    load_dir = None
    train_file = None
    if dataset == "jigsaw-1":
        load_dir = base_path + "jigsaw-toxic-comment-classification-challenge/"
        train_file = load_dir + "train.csv"

    if load_dir:
        if train_file:
            dataset = slicesFromPanda(train_file, dataset)

            # dataset = datasetFromTensor(train_file, batch_size, buffer_size)
            # print(tokenizeDataset(dataset))
            # return tokenizeDataset(dataset)
            return dataset
    else:
        print("Not a valid dataset!")
    return None


# if __name__ == "__main__":
#     getDataset("jigsaw-1")
#     # readDatset("datasets/jigsaw-toxic-comment-classification-challenge/train.csv")
