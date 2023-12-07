import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models

labels = ['A','B','C','D','E','F','G','H']
all_train_images = []
all_train_labels = []

for label in labels:

    image_paths = [f'Data_Collected\{label}\Image_{i}.jpg' for i in range(1,401)]
    # Reading Images
    load_images = [cv2.imread(path) for path in image_paths]
    # Normalize pixel values
    set_images = [img.astype('float32') / 255 for img in load_images]

    all_train_images.extend(set_images)
    
    # Assign labels to the images
    set_labels = [labels.index(label)] * len(set_images)
    all_train_labels.extend(set_labels)

# Convert to NumPy arrays
train_images = np.array(all_train_images)
train_labels = np.array(all_train_labels)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())  # Flatten the spatial dimensions

# More dense layers can follow
model.add(layers.Dense(8, activation='softmax'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

num_epochs = iterations = 10
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=num_epochs, validation_split=0.2)
model.save('Model\keras_model.h5')
print(model.predict('Data_Collected\A\Image_1.jpg'))
