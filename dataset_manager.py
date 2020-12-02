import pandas as pd
from pandas import DataFrame
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
