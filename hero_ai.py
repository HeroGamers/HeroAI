import tensorflow_datasets as tfds
import tensorflow as tf

# Variables for the AI
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_STEPS = 30
LEARNING_RATE = 1e-4


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder


train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)


# We make a function for the model creation, just to beautify the code c:
def create_model():
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
def train(model, training_data, testing_data):
    # We train the model
    history = model.fit(training_data, epochs=EPOCHS,
                        validation_data=testing_data,
                        validation_steps=VALIDATION_STEPS)


# Create the model
model = create_model()
# Get model summary
model.summary()
# We train the model
train(model, train_dataset, test_dataset)


# We test the model
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
