# -*- coding: utf-8 -*-
"""

Challenge: The CIFAR-10 dataset
Note: This is a notoriously difficult dataset to learn using fully connected neural networks. 
Let's see how well we can learn it!

Ion Petre FoundML_course_assignments
/FML2w4_deep_learning.ipynb
https://github.com/ionpetre/FoundML_course_assignments/blob/main/FML2w4_deep_learning.ipynb

@author: jarno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from keras.datasets import cifar10
from keras.utils import to_categorical

from keras import models
from keras import layers
import random

(X_train_valid, y_train_valid), (X_test, y_test) = cifar10.load_data()

print('We have %2d training pictures and %2d test pictures.' % (X_train_valid.shape[0],X_test.shape[0]))
print('Each picture is of size (%2d,%2d,%2d)' % (X_train_valid.shape[1], X_train_valid.shape[2], X_train_valid.shape[3]))


# Scale the data into [0,1] by dividing to 255

X_train_valid_std = X_train_valid/255
X_test_std  = X_test/255


# Display some images

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


plt.figure(figsize=(20,12))
for i in range(15): # i start from zero
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j = random.randint(2,40)
    plt.imshow(X_train_valid_std[(i+1)*j])
    plt.xlabel(class_names[int(y_train_valid[(i+1)*j])], fontsize=30)
plt.show()

# Is the dataset balanced?

y_train_valid_count = np.unique(y_train_valid, return_counts=True)
df_y_train_valid = pd.DataFrame({'Label':y_train_valid_count[0], 'Count':y_train_valid_count[1]})
df_y_train_valid

# A: YES!
# Train - validation split 

X_train_std, X_valid_std, y_train, y_valid = train_test_split(
    X_train_valid_std, 
    y_train_valid, 
    test_size=0.2, #validation size 
    random_state=150, 
    stratify=y_train_valid,
    shuffle=True
)

# Check the result of the data split

print('# of training images:', X_train_std.shape[0])
print('# of validation images:', X_valid_std.shape[0])
print("Note the shape of the data (3 color channels):", X_train_std.shape)


# Encode the labels from numerical to categorical

#from keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=10)
y_valid_cat = to_categorical(y_valid, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)


# Train an ANN model with an input "Flatten" layer of shape (32, 32, 3), accounting for the 3 color channels,
#       followed by 3 layers of size 128/64/32, followed by an output layer of a suitable size.
# Choose 'relu' for the activation function of the hidden layers, and a suitable activation for the output layer. 


ANNmodel = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax'),
])





# Compile the model using the Adam optimizer with learning rate 1e-3 
#       and as metrics CategoricalAccuracy and TruePositives.
# Use as the loss function CategoricalCrossentropy()



ANNmodel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(), 
             tf.keras.metrics.TruePositives(),
            ],
)

ANNmodel.summary()




# We reset all variables implicitly instantiated by Keras/tensorflow
tf.keras.backend.clear_session()

# This callback will stop the training when there is no improvement in the loss 
#      for ten consecutive epochs.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# Fit the model by specifying the number of epochs and the batch size
# We also indicate the validation data so we can collect the evolution 
#      of the metrics through the epochs, both on train, as well as on validation.

ANN_fit_history = ANNmodel.fit(X_train_std,
                               y_train_cat, 
                               epochs=300, 
                               batch_size=128,
                               callbacks=[callback],
                               validation_data=(X_valid_std, y_valid_cat)
                              )



history_dict = ANN_fit_history.history
print(history_dict.keys())

# Plot the evolution of the loss and the accurayc throughout the epochs
# This is useful to find over-fitting and decide on early stopping of the training. 

#import matplotlib.pyplot as plt

train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['categorical_accuracy']
val_acc = history_dict['val_categorical_accuracy']
train_tp = np.array(history_dict['true_positives']) / X_train_std.shape[0]
val_tp = np.array(history_dict['val_true_positives']) / X_valid_std.shape[0]
epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(20, 5))

plt.subplot(1,3,1)
plt.plot(epochs, train_loss, 'b', label='Training cat. cross-entropy')
plt.plot(epochs, val_loss, 'r', label='Validation cat. cross-entropy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1,3,2)
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Categorical accuracy')
plt.legend()

plt.subplot(1,3,3)
plt.plot(epochs, train_tp, 'b', label='Training TP')
plt.plot(epochs, val_tp, 'r', label='Validation TP')
plt.title('Training and validation true positives')
plt.xlabel('Epochs')
plt.ylabel('True positives')
plt.legend()


plt.show()





y_train_prob = ANNmodel.predict(X_train_std)

# Select the most likely class
y_train_pred=np.argmax(y_train_prob, axis=1)

print("\n The classification results on the train data:")
print(classification_report(y_train,y_train_pred))
print("Confusion matrix (train data):\n", confusion_matrix(y_train,y_train_pred))


# The classification results for the validation data

y_valid_prob = ANNmodel.predict(X_valid_std)
y_valid_pred=np.argmax(y_valid_prob, axis=1)
print("\n The classification results on the validation data:")
print(classification_report(y_valid,y_valid_pred))
print("Confusion matrix (validation data):\n", confusion_matrix(y_valid,y_valid_pred))
print("Confusion matrix (validation data):\n", np.round(confusion_matrix(y_valid,y_valid_pred)/10))
print("Confusion matrix (validation data):\n", np.round(confusion_matrix(y_valid,y_valid_pred)/10).astype(int))



# Plot the first X validation images, their predicted labels, and the true labels in parenthesis.
# Color correct predictions in blue and incorrect predictions in red.


def plot_image(i, predictions_array, true_label, img):
    true_label, img = int(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color, fontsize=10, loc='left')

def plot_value_array(i, predictions_array, true_label):
    true_label = int(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')




num_rows = 3
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(3*2*num_cols, 2*num_rows))

for i in range(num_images):
    j = random.randint(4, 20)
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i*j, y_valid_prob[i*j], y_valid, X_valid_std)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i*j, y_valid_prob[i*j], y_valid)
    plt.xticks(range(10), class_names, rotation=90)
    
plt.tight_layout()
plt.show()


