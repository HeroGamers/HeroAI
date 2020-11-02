import tensorflow_datasets as tfds
import tensorflow as tf
from os import path, environ
from datetime import datetime
import dataset_manager as dsmg

# Fix for finding the dnn implementation
environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Variables for the AI learning
BUFFER_SIZE = 10000  # Used for suffling the datasets
BATCH_SIZE = 64  # Samples we run through before the model is updated
EPOCHS = 5  # The amount of times the dataset is run through in training
VALIDATION_STEPS = 30  # How many batches we run through during validation
LEARNING_RATE = 1e-4  # The rate at which our weights will be updated
TAKE_SIZE = 100  # How much we take from the dataset
EMBED_DIM = 64  # The size of the embed layer's vector space
DEEP_UNITS = 64  # The amount of units in our deep network LSTM layer
DENSE_UNITS = 64  # The amount of units in our dense layer
DATA_MAX_LENGTH = 2000


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
        dataset, word_vectors, input_max_length = dsmg.getDataset(dataset)

        print("Max length: " + str(input_max_length))

        train_data = dataset.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
        train_data = train_data.padded_batch(BATCH_SIZE)

        test_data = dataset.take(TAKE_SIZE)
        test_data = test_data.padded_batch(BATCH_SIZE)

        return train_data, test_data, word_vectors + 1, input_max_length


# We make a function for assigning a checkpoint-file to use for training
def new_checkpointfile(name=None):
    base_dir = "training_models/"
    if not name:
        i = 1
        while True:
            current_path = str(base_dir + "cp_" + str(i) + ".ckpt")
            if path.exists(current_path):
                i += 1
            else:
                checkpoint_path = current_path
                break
    else:
        checkpoint_path = base_dir + str(name) + ".cpkt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    return cp_callback


# Function for saving the current model - either entire model or just weights
def save_model(model, deep=False, name=None):
    base_dir = "models/"
    if deep:
        base_dir = base_dir + "deep_models/"
    else:
        base_dir = base_dir + "weight_models/"

    # We use the current time as the name for the saved model, if no name has been given
    if name:
        if deep:
            current_path = base_dir + str(name)
            if not path.exists(current_path):
                model.save(current_path)
                return True
        else:
            current_path = base_dir + str(name) + ".ckpt"
            if not path.exists(current_path):
                model.save_weights(current_path)
                return True
        return False
    else:
        time = datetime.now()
        time_str = time.strftime("%Y%m%d%H%M%S")
        if deep:
            current_path = base_dir + time_str
            if not path.exists(current_path):
                model.save(current_path)
                return True
        else:
            current_path = base_dir + time_str + ".ckpt"
            if not path.exists(current_path):
                model.save_weights(current_path)
                return True
        return False


# We make a function for the model creation, just to beautify the code c:
def create_model(vocab_size):
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBED_DIM),  # Embedding layer to store one vector pr. word
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
def train(model, training_data, testing_data, callbacks=None):
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
    dataset = "jigsaw-1"
    print("Loading the " + dataset + " dataset...")
    train_dataset, test_dataset, vocab_size, input_max_length = load_dataset(dataset, False)
    print("Done loading!")

    # Create the model
    print("Creating the model...")
    model = create_model(vocab_size)
    print("Model created!")

    # Get model summary
    model.summary()

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

    # We test the model
    print("Testing the model...")
    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print("Done testing the model!\n")

    return model


# Run HeroAI
run()
