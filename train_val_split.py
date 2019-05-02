import os
import numpy as np
import shutil

np.random.seed(42)

path_train = 'data/train_split'
path_val = 'data/val_split'
path_total = 'data/train'

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

num_train = 0
num_val = 0

# train test split
split = 0.8

for fish in classes:
    if fish not in os.listdir(path_train):
        os.mkdir(os.path.join(path_train, fish))

    total_images = os.listdir(os.path.join(path_total, fish))

    num_train_img = int(len(total_images) * split)

    np.random.shuffle(total_images)

    train_images = total_images[:num_train_img]

    val_images = total_images[num_train_img:]

    for img in train_images:
        source = os.path.join(path_total, fish, img)
        target = os.path.join(path_train, fish, img)
        shutil.copy(source, target)
        num_train += 1

    if fish not in os.listdir(path_val):
        os.mkdir(os.path.join(path_val, fish))

    for img in val_images:
        source = os.path.join(path_total, fish, img)
        target = os.path.join(path_val, fish, img)
        shutil.copy(source, target)
        num_val += 1

print('# numbers of train samples: {}, # numbers of val samples: {}'.format(num_train, num_val))
