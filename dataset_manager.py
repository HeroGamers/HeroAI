import pandas as pd
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


def slicesFromTensor(file, batch_size):
    dataset_slices = tf.data.experimental.make_csv_dataset(file, batch_size=batch_size, label_name="toxic",
                                                           select_columns=["comment_text", "toxic"])
    for feature_batch, label_batch in dataset_slices.take(1):
        print("Toxic: {}".format(label_batch))
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))
    return dataset_slices


def getDataset(dataset):
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
            slice = slicesFromTensor(train_file, 5)
    else:
        print("Not a valid dataset!")
    return None


if __name__ == "__main__":
    getDataset("jigsaw-1")
    # readDatset("datasets/jigsaw-toxic-comment-classification-challenge/train.csv")
