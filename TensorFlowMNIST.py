import tensorflow as tf
# Display this TensorFlow version
#print("TensorFlow Version:",tf.__version__)

# Download mnist set (if not already downloaded)
mnist=tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)=mnist.load_data()
x_train, x_test=x_train/255.0,x_test/255.0

# Flatten takes an nxm (28x28) and "flattens" the input for 128 input nodes
# Dense specifies using a Rectified Linear Unit activation function for 128 nodes
# Dropout ?
# Dense specifies 10 output nodes to represent values 0 through 9
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Process the results of the training data with the untrained model
predictions = model(x_train[:1]).numpy()

# Convert logits/log-odds scores into probabilities
tf.nn.softmax(predictions).numpy()

# Define a loss function for training, note that the initial (untrained)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Compile the model, using the previously defined loss function
model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])

# Train the module using the training data
model.fit(x_train, y_train, epochs=5)

# Determine the suitability by processing different, non-training data
model.evaluate(x_test,  y_test, verbose=2)

#probability_model = tf.keras.Sequential([
#    model,
#    tf.keras.layers.Softmax()
#])

#probability_model(x_test[:5])
