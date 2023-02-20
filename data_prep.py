import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
tf.config.experimental.list_physical_devices()

#%%
tf.test.is_built_with_cuda()

#%%
image_size = 227
batch_size = 32
color = 1           # RGB
epochs = 50

dataset = tf.keras.utils.image_dataset_from_directory(
    "images",
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size, image_size),
)

class_names = dataset.class_names

#%%
def process_image(file):
    label = tf.strings.split(file, os.path.sep)[-1]
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [image_size, image_size])
    return label, img

file = "images/angry/0.jpg"
label, img = process_image(file)
img = img/255
plt.imshow(img, cmap="gray")


#%% print info
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)  # batch_size, image_size, image_size, color (RGB)
    print(labels_batch.shape)
    break

#%% visualize image
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.title(class_names[labels_batch[i]])
        plt.imshow(image_batch[i].numpy().astype("uint8"), cmap="gray")
        plt.axis("off")
        plt.show()

#%%
for image_batch, labels_batch in dataset.take(1):
    print(image_batch[0])
    print(labels_batch[0])
    print(image_batch[0].shape)

#%% split dataset into train and test
# 80% train, 10% validation, 10% test
def get_dataset_partition_tf(ds, train_split=.8, val_split=.1, shuffle=True, shuffle_size=10):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12345)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)    #keep first 80%
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)


#%%
# cache --> read img from disk, from next interation keep the img in the memory
# prefetch --> load the next batch while the current batch is being processed
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#%% preprocessing --> scale
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(image_size, image_size),
    layers.experimental.preprocessing.Rescaling(1.0/255),
])

# data augmentation --> create new samples with bit different characteristics
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

#%% MODEL TRAINING -- Convolutional neural network
image_size = 48
batch_size = 64
color = 1           # RGB
epochs = 50

n_classes = 7
n_epochs = 40
learning_rate = 0.01
input_shape = (batch_size, image_size, image_size, color)

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(filters=32,               # layers --> try and errors
                  kernel_size=(3, 3),       # kernel_size = eg 3X3 filter
                  activation='relu',        # activation function always relu
                  input_shape=input_shape), # input_shape = (batch_size, image_size, image_size, color)
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=128,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=512,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),   # 64 neurons
    layers.Dense(n_classes, activation='softmax')  # softmax --> probability
])

model.build(input_shape=input_shape)
model.summary()



#%%
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

#%%
history = model.fit(
    train_ds,
    epochs=n_epochs,
    batch_size=batch_size,
    verbose=1,
    validation_data=val_ds
)

#%%
scores = model.evaluate(test_ds)
print(scores)

#%%
history.history.keys()

#%%
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(n_epochs), accuracy, label='Training Accuracy')
plt.plot(range(n_epochs), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(n_epochs), loss, label='Training Loss')
plt.plot(range(n_epochs), val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')

plt.show()
fig.savefig('plot.png')
#%%

def print_prediction(batch, image_index):
    for image_batch, labels_batch in test_ds.take(batch):
        img = image_batch[image_index].numpy().astype("uint8")
        lbl = labels_batch[image_index]

        print("first image to predict")
        plt.imshow(img, cmap="gray")
        print('label: ', lbl)

        # find predicted label
        batch_prediction = model.predict(image_batch)
        print(batch_prediction[image_index])        ## RETURNS PROBABILITY FOR EACH CLASS
        print(class_names[np.argmax(batch_prediction[image_index])])

print_prediction(1, 5)
#%%
def get_prediction_confidence(batch, image_index):
    for image_batch, labels_batch in test_ds.take(batch):
        img = image_batch[image_index].numpy().astype("uint8")
        lbl = labels_batch[image_index]

        # find predicted label
        batch_prediction = model.predict(image_batch)
        confidence = batch_prediction[image_index]        ## RETURNS PROBABILITY FOR EACH CLASS
        print(class_names[np.argmax(batch_prediction[image_index])])

        return class_names, confidence

x = get_prediction_confidence(1, 15)
print(x)

#%%
model_version="model2"
model.save(f"models/m3_H5", save_format='h5')
tf.keras.models.save_model(model, f"models/{model_version}_TF", save_format='h5')

#%%
save_ck = callbacks.ModelCheckpoint()