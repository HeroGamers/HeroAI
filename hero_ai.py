import tensorflow_datasets as tfds
import tensorflow as tf
from os import path
from datetime import datetime

# Variables for the AI learning
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_STEPS = 30
LEARNING_RATE = 1e-4


# Function for loading a dataset to use on the model
def load_dataset(dataset, using_tfds=True):
    if using_tfds:
        dataset, info = tfds.load(dataset, with_info=True,
                                  as_supervised=True)
        train_dataset, test_dataset = dataset["train"], dataset["test"]

        encoder = info.features["text"].encoder

        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.padded_batch(BATCH_SIZE)
        test_dataset = test_dataset.padded_batch(BATCH_SIZE)

        return train_dataset, test_dataset, encoder


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
def create_model(encoder):
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),  # Embedding layer to store one vector pr. word
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # Bidirectional layer for RNN
        tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with neurons
        tf.keras.layers.Dropout(0.5),  # We put a dropout layer to prevent overfitting
        tf.keras.layers.Dense(1)  # Single output layer for the toxicity
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


# Function to test the model
def test_model(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    return test_loss, test_acc


def run():
    # Load the dataset
    # datasets for testin' include "imdb_reviews/subwords8k"
    dataset = "imdb_reviews/subwords8k"
    print("Loading the " + dataset + " dataset...")
    train_dataset, test_dataset, encoder = load_dataset(dataset)
    print("Done loading!")

    # Create the model
    print("Creating the model...")
    model = create_model(encoder)
    print("Model created!")

    # Get model summary
    model.summary()

    # Get a new checkpoint file and callback to it
    print("Readying the checkpoint file...")
    cp_callback = new_checkpointfile()
    callbacks = [cp_callback]
    print("Checkpoint file ready!")

    # We train the model
    print("Initiating training sequence...")
    history = train(model, train_dataset, test_dataset, callbacks)
    print("Training finished!")

    # We save the model
    print("Saving the model...")
    saved = save_model(model, False)
    if saved:
        print("Saved the model!")
    else:
        print("An error occurred whilst saving the model!")

    # We test the model
    print("Testing the model...")
    test_loss, test_acc = test_model(model, test_dataset)
    print("Done testing the model!\n")

    return model


# Run HeroAI
run()
