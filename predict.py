#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import math

weights_path = "./data/weights"
test_data_path = "./data/test_stg1/"
batch_size = 32
test_samples_nums = 1000

test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

InceptionV3_model = load_model(weights_path)

# aug 5 times to feed prediction
for i in range(5):
    test_generator = test_datagen.flow_from_directory(
            test_data_path,
            target_size=(299, 299),
            batch_size=batch_size,
            shuffle=False,
            classes=None,
            class_mode=None)
    images_list = test_generator.filenames
    if i == 0:
        pred_res = InceptionV3_model.predict_generator(test_generator, math.ceil(test_samples_nums/batch_size))
    else:
        pred_res += InceptionV3_model.predict_generator(test_generator, math.ceil(test_samples_nums/batch_size))

pred_res /= 5


# write submit file
with open('./data/submits/submit.csv', 'w') as submit:
    submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    for i, image_name in enumerate(images_list):
        prob = ['%.6f' % digit for digit in pred_res[i, :]]
        submit.write('%s,%s\n' % (image_name, ',' + str(prob)))

