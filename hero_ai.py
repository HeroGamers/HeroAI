import tensorflow_datasets as tfds
import tensorflow as tf
import io
from os import path, environ, makedirs
from datetime import datetime
import dataset_manager as dsmg

# Fix for finding the dnn implementation
environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Variables for the AI learning
BUFFER_SIZE = 10000  # Used for suffling the datasets
BATCH_SIZE = 64  # Samples we run through before the model is updated
EPOCHS = 15  # The amount of times the dataset is run through in training
VALIDATION_STEPS = 30  # How many batches we run through during validation
LEARNING_RATE = 1e-4  # The rate at which our weights will be updated
EMBED_DIM = 64  # The size of the embed layer's vector space
DEEP_UNITS = 64  # The amount of units in our deep network LSTM layer
DENSE_UNITS = 64  # The amount of units in our dense layer

# Data variables - also changes learning
MAX_FEATURES = 10000  # The max vocab size we will train
MAX_LENGTH = 2000  # The max number of words in a sentence we will take
TRAIN_TAKE_SIZE = 0  # How much we take from the dataset for the training - can be set to 0 to take everything (needs to be at least BATCH_SIZE. Steps for each epoch is TRAIN_TAKE_SIZE//BATCH_SIZE)
TEST_TAKE_SIZE = 3000  # How much we take from the dataset for the test and validation (needs to be at least VALIDATION_STEPS*BATCH_SIZE)


# Function for loading a dataset to use on the model
def load_dataset(dataset, using_tfds=True):
    if using_tfds:
        dataset, info = tfds.load(dataset, with_info=True,
                                  as_supervised=True)
        train_dataset, test_dataset = dataset["train"], dataset["test"]

        encoder = info.features["text"].encoder

        if not encoder:
            return None

        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.padded_batch(BATCH_SIZE)
        test_dataset = test_dataset.padded_batch(BATCH_SIZE)

        return train_dataset, test_dataset, encoder.vocab_size, None
    else:
        dataset = dsmg.getDataset(dataset)

        # We take all the text from the dataset, and put it into one set - for the encoder
        text_data = dataset.map(lambda text, label: text)

        # ---=Time to get the training dataset and the testing dataset=--- #

        # Data performance - prefetching (https://www.tensorflow.org/guide/data_performance#prefetching)
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # We skip the given amount of size for the test set, and shuffle
        train_data = dataset.skip(TEST_TAKE_SIZE).shuffle(BUFFER_SIZE)
        # If we are given a training size, take that
        if TRAIN_TAKE_SIZE > 0:
            train_data = train_data.take(TRAIN_TAKE_SIZE)
        # Create batches of our batch size, cache the dataset to memory, and put prefetching on it
        train_data = train_data.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

        # For the test set, we take the test size, that got skipped before, create batches of our batch size,
        # cache and put prefetching on it
        test_data = dataset.take(TEST_TAKE_SIZE).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

        return text_data, train_data, test_data


# We make a function for assigning a checkpoint-file to use for training
def new_checkpointfile(name=None):
    base_dir = "training_models/"
    if not name:
        i = 1
        while True:
            current_path = str(base_dir + "cp_" + str(i) + ".ckpt")
            if path.exists(current_path+".index"):
                i += 1
            else:
                checkpoint_path = current_path
                break
    else:
        checkpoint_path = base_dir + str(name) + ".cpkt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    return cp_callback


# Function for saving the current model - either entire model or just weights
def save_model(model: tf.keras.Model, deep=False, embedding=None, name=None):
    def save_deep(base_dir, name):
        current_path = base_dir + str(name)
        if not path.exists(current_path):
            model.save(current_path)
            return True
        return False

    def save_weights(base_dir, name):
        current_path = base_dir + str(name) + ".ckpt"
        if not path.exists(current_path):
            model.save_weights(current_path)
            return True
        return False

    def save_embedding(base_dir, name, encoder):
        current_path = base_dir + str(name)
        if not path.exists(current_path):
            # Check if the file locations exist
            if not path.exists(current_path):
                makedirs(current_path)

            # From https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_trained_word_embeddings_and_save_them_to_disk
            weights = model.get_layer('embedding').get_weights()[0]
            vocab = encoder.get_vocabulary()

            out_v = io.open(current_path+'/vectors.tsv', 'w', encoding='utf-8')
            out_m = io.open(current_path+'/metadata.tsv', 'w', encoding='utf-8')

            for index, word in enumerate(vocab):
                if index == 0:
                    continue  # skip 0, it's padding.
                vec = weights[index]
                out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                out_m.write(word + "\n")
            out_v.close()
            out_m.close()
            return True

    # Let the saving begin
    base_dir = "models/"

    # We use the current time as the name for the saved model, if no name has been given
    if not name:
        time = datetime.now()
        name = time.strftime("%Y%m%d%H%M%S")

    if embedding:
        base_dir = base_dir + "embedding_projector/"
        return save_embedding(base_dir, name, embedding)
    else:
        if deep:
            base_dir = base_dir + "deep_models/"
            return save_deep(base_dir, name)
        else:
            base_dir = base_dir + "weight_models/"
            return save_weights(base_dir, name)


# Function for text encoder
def create_encoder(dataset):
    # Then we create the encoding layer
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=MAX_FEATURES,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=MAX_LENGTH)

    # Fit the encoder layer
    encoder.adapt(dataset)

    return encoder


# We make a function for the model creation, just to beautify the code c:
def create_model(encoder):
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),  # Our input into the model is a string
        encoder,  # Our text encoder to make text into integers
        tf.keras.layers.Embedding(input_dim=MAX_FEATURES, output_dim=EMBED_DIM, mask_zero=True),  # Embedding layer to store one vector pr. word integer
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(DEEP_UNITS)),  # Bidirectional layer for RNN
        tf.keras.layers.Dense(DENSE_UNITS, activation="relu"),  # Dense layer with neurons
        tf.keras.layers.Dropout(0.5),  # We put a dropout layer to prevent overfitting
        tf.keras.layers.Dense(1, activation="sigmoid")  # Single output layer for the toxicity, sigmoid for probability
    ])

    # Compile the Keras model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Our loss function, binary cuz two labels
                  optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),  # Our loss optimizer - we use Adam
                  metrics=['accuracy'])  # Metric we use to monitor our loss with

    return model


# A function for training
def train(model: tf.keras.Model, training_data, testing_data, callbacks=None):
    # Check for checkpoint
    if callbacks:
        history = model.fit(training_data, epochs=EPOCHS,
                            validation_data=testing_data,
                            validation_steps=VALIDATION_STEPS,
                            callbacks=callbacks)
    else:
        history = model.fit(training_data, epochs=EPOCHS,
                            validation_data=testing_data,
                            validation_steps=VALIDATION_STEPS)

    return history


def run():
    # Load the dataset
    # datasets which the AI can run from tensorflow_datasets include "wikipedia_toxicity_subtypes"
    dataset_name = "jigsaw-1"
    print("Loading the " + dataset_name + " dataset...")
    text_dataset, train_dataset, test_dataset = load_dataset(dataset_name, False)
    print("Done loading!")

    # Create the encoder
    print("Creating the encoder...")
    encoder = create_encoder(text_dataset)
    print("Encoder created!")
    print("Encoder vocab: " + str(len(encoder.get_vocabulary())))

    # Create the model
    print("Creating the model...")
    model = create_model(encoder)
    print("Model created!")

    # Get model summary
    try:
        model.summary()
    except ValueError as e:
        print("Couldn't get model summary - " + str(e))

    # Get a new checkpoint file and callback to it
    print("Readying the checkpoint file...")
    cp_callback = new_checkpointfile()
    time = datetime.now()
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/training/" + time.strftime("%Y%m%d%H%M%S"),
                                                 histogram_freq=1)
    callbacks = [cp_callback, tb_callback]
    print("Checkpoint file ready!")

    # We train the model
    print("Initiating training sequence...")
    history = train(model, train_dataset, test_dataset, callbacks)
    print("Training finished!")

    # We save the model
    print("Saving the model, weights only...")
    saved = save_model(model, False)
    if saved:
        print("Saved the model!")
    else:
        print("An error occurred whilst saving the model!")

    print("Saving the model, everything...")
    saved = save_model(model, True)
    if saved:
        print("Saved the model!")
    else:
        print("An error occurred whilst saving the model!")

    print("Saving the model, trained word embeddings...")
    saved = save_model(model, embedding=encoder)
    if saved:
        print("Saved the model!")
    else:
        print("An error occurred whilst saving the model!")

    # We test the model
    print("Testing the model...")
    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print("Done testing the model!\n")

    return model


# Run HeroAI
model = run()
