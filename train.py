#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

num_train_samples = 3019
num_validation_samples = 758
epochs = 25
batch_size = 32


train_data_dir = 'data/train_split'
val_data_dir = 'data/val_split'
best_model_file = 'data/weights.h5'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def InceptionV3_Model(input_shape=(299, 299, 3), learning_rate=0.0001):

    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                            input_tensor=None, input_shape=input_shape)

    output = InceptionV3_notop.get_layer(index=-1).output  # Shape=(8, 8, 2048)
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(8, activation='softmax', name='predictions')(output)
    model_v3 = Model(InceptionV3_notop.input, output)
    optimizer = Adam(lr=learning_rate)
    model_v3.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model_v3



def train_generator(classes, batch_size):
    train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
    generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(299, 299),
                batch_size=batch_size,
                shuffle=True,
                classes=classes,
                class_mode='categorical')
    return generator



def val_generator(classes, batch_size):
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
                val_data_dir,
                target_size=(299, 299),
                batch_size=batch_size,
                shuffle=True,
                classes=classes,
                class_mode='categorical')
    return validation_generator


def main():
    inception_v3 = InceptionV3_Model()
    tr_generator = train_generator(classes=classes, batch_size=batch_size)
    validation_generator = val_generator(classes=classes, batch_size=batch_size)
    checkpoint = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)
    # model fit
    inception_v3.fit_generator(
            tr_generator,
            samples_per_epoch=num_train_samples,
            nb_epoch=epochs,
            validation_data=validation_generator,
            nb_val_samples=num_validation_samples,
            callbacks=[checkpoint])



if __name__ == '__main__':
    main()
