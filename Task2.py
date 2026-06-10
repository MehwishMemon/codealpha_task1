import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models

# ------------------------
# 1. Load Dataset (MNIST)
# ------------------------

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ------------------------
# 2. Build CNN Model
# ------------------------

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# ------------------------
# 3. Compile Model
# ------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ------------------------
# 4. Train Model
# ------------------------

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)


# ------------------------
# 5. Evaluate Model
# ------------------------

loss, acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", acc)


# ------------------------
# 6. Save Model
# ------------------------

model.save("handwriting_model.h5")

print("\nModel Saved Successfully")


# ------------------------
# 6. Make Predictions
# ------------------------

# Take one test image
index = 7  # change this to test different images

sample_image = x_test[index]

# reshape for model (1 image)
sample_image = np.expand_dims(sample_image, axis=0)

# predict
prediction = model.predict(sample_image)

# get class with highest probability
predicted_class = np.argmax(prediction)

plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted: {predicted_class}, Actual: {y_test[index]}")
plt.show()

print("\nPredicted Digit:", predicted_class)
print("Actual Digit:", y_test[index])