from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import sklearn.metrics as metrics

import time
start_time = time.time()


train_data_path = 'data/train'
validation_data_path = 'data/test'
test_data_path = 'data/alien_test'

img_width, img_height = 150, 150
batch_size = 32 

# Import the inception model

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  #Formato (150x150, 3 canais) 
                                include_top=False, #exclui a ultima camada do modelo pre treinado
                                weights='imagenet')


for layer in pre_trained_model.layers:
    layer.trainable = False


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.959):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop


x = layers.Flatten()(pre_trained_model.output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x) #dropout para reduzir overfitting                 
#saida
x = layers.Dense  (5, activation='softmax')(x) # ativação softmax para classificação multi classe 

model = Model( pre_trained_model.input, x) 

model.summary()


model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')


callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            #steps_per_epoch = int(1785/batch_size),
            epochs = 100,
            #validation_steps = int(598/batch_size),
            verbose = 2,
            callbacks=[callbacks])


predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
#print(len(predicted_classes))
print(predicted_classes)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print(true_classes)
#print(testdata)
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)    
confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
print(confusion_matrix) 
print("Accuracy: ", metrics.accuracy_score(true_classes, predicted_classes))
print("Precision: ", metrics.precision_score(true_classes, predicted_classes, average='micro'))

print("--- %s seconds ---" % (time.time() - start_time))


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b