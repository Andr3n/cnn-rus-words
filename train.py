import numpy as np
import os
import random

import tensorflow as tf

from pathlib import Path
from keras.models import Sequential
from PIL import Image, ImageChops, ImageFilter
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.utils import np_utils

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101

def augmentation(img, method, params=[]):
    try:
        lenght = len(params)
    except:
        raise('Wrong parameters')
    
    if method == 'rotate':
        if len(params) == 0:
            params = (-30, 30)
        elif len(params) == 1:
            params = (0, params[0])
            
        res_img = Image.new("RGB", img.size, (255, 255, 255))
        res_img.paste(img)
        angle = random.randint(params[0], params[1])
        res_img = res_img.rotate(angle, fillcolor='white')

    elif method == 'blur':
        if len(params) == 0:
            params = (1.2,)
        elif len(params) > 1:
            raise('Wrong parameters')
        
        res_img = Image.new("RGB", img.size, (255, 255, 255))
        res_img.paste(img)
        res_img = res_img.filter(filter=ImageFilter.GaussianBlur(params[0]))

    elif method == 'shift':
        res_img = Image.new("RGB", img.size, (255, 255, 255))
        res_img.paste(img)
        horizontal, vertical = random.randint(-30, 30), random.randint(-30, 30)
        res_img = ImageChops.offset(res_img, horizontal, vertical)
    else:
        res_img = img
    return res_img

# Без увеличения данных
img_path = Path(r'Path-to-dataset-folder')

MAX_SIZE = (270, 115)
MIN_SIZE = (90, 40)
MEAN_SIZE = (int(np.mean([MIN_SIZE[0], MAX_SIZE[0]])), int(np.mean([MIN_SIZE[1], MAX_SIZE[1]])))

words_list = ['и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'этот', 'к', 'но', 'они', 'мы', 'тут', 'из', 'у', 'который', 'то', 'за', 'свой',
 'сейчас', 'весь', 'год', 'от', 'так', 'о', 'для', 'ты', 'же', 'все', 'тот', 'мочь', 'вы', 'человек', 'такой', 'его', 'сказать', 'только', 'или', 'ещё', 'бы', 'себя', 'один', 'как', 'уже',
 'до', 'время', 'если', 'сам', 'когда', 'другой', 'вот', 'говорить', 'наш', 'мой', 'знать', 'стать', 'при', 'чтобы', 'дело', 'жизнь', 'кто', 'первый', 'очень', 'два', 'день', 'её', 'новый', 'рука', 'даже',
 'во', 'со', 'раз', 'где', 'там', 'под', 'можно', 'ну', 'какой', 'после', 'их', 'работа', 'без', 'самый', 'потом', 'надо', 'хотеть', 'ли', 'слово', 'идти', 'большой', 'должен', 'место', 'иметь', 'ничто']

words_list_en = ['i', 'v', 'ne', 'na', 'ja', "byt'", 'on', 's', 'chto', 'a', 'po', 'eto', 'ona', 'etot', 'k', 'no', 'oni', 'my', 'tut', 'iz', 'u', 'kotoryj', 'to', 'za', 'svoj',
 'sejchas', "ves'", 'god', 'ot', 'tak', 'o', 'dlja', 'ty', 'zhe', 'vse', 'tot', "moch'", 'vy', 'chelovek', 'takoj', 'ego', "skazat'", "tol'ko", 'ili', 'esche', 'by', 'sebja', 'odin', 'kak', 'uzhe',
 'do', 'vremja', 'esli', 'sam', 'kogda', 'drugoj', 'vot', "govorit'", 'nash', 'moj', "znat'", "stat'", 'pri', 'chtoby', 'delo', "zhizn'", 'kto', 'pervyj', "ochen'", 'dva', "den'", 'ee', 'novyj', 'ruka', 'dazhe',
 'vo', 'so', 'raz', 'gde', 'tam', 'pod', 'mozhno', 'nu', 'kakoj', 'posle', 'ih', 'rabota', 'bez', 'samyj', 'potom', 'nado', "hotet'", 'li', 'slovo', 'idti', "bol'shoj", 'dolzhen', 'mesto', "imet'", 'nichto']

def get_data_train(classes_count, include_augmentation=True):
    # RGB
    img_path_train = img_path / 'train'
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    imgs = []
    augmentation_ways = ['original',] #  'shift', 'rotate',  'blur'
    
    if include_augmentation:
        augmentation_ways = ['original', 'blur', 'rotate']
    
    
    class_dot = 0
    for word in words_list_en[:classes_count]:
        index = 0
        print(word, class_dot)

        for word_path in img_path_train.glob(f'{word}_*'):
            img = Image.open(word_path)

            for way in augmentation_ways:

                res_img = augmentation(img, way)

                res_img = res_img.resize(MIN_SIZE)    
                img_arr = np.array(res_img)

                if index < 90:
                    X_train.append(img_arr)
                    y_train.append([class_dot])
                else:
                    imgs.append(res_img)
                    X_test.append(img_arr)
                    y_test.append([class_dot])

            index += 1
        class_dot += 1

    X_train = np.array(X_train, dtype='float32')
    y_train = np.array(y_train, dtype='uint8')
    X_val = np.array(X_test, dtype='float32')
    y_val = np.array(y_test, dtype='uint8')

    # Преобразуем метки классов в категории
    Y_train = np_utils.to_categorical(y_train, classes_count)
    Y_val = np_utils.to_categorical(y_val, classes_count)

    # Normaliz
    X_train = X_train / 255
    X_val = X_val / 255
    
    return X_train, Y_train, X_val, Y_val


optimizer='SGD'

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

name = 'VGG19'
epochs = 300 #4k
bs=8
classes_count = 100

total_epochs = 0
X_train, Y_train, X_val, Y_val = get_data_train(classes_count, include_val=True)

global_path = f'./result/model_{name}/chk_points/'
checkpoint_path = global_path + '/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, 
                    verbose=1, 
                    save_weights_only=True)


model = Sequential()
model.add(VGG16(include_top=False, input_shape=(MIN_SIZE[1], MIN_SIZE[0], 3))) # SGD

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
             metrics=['accuracy',  precision_m, recall_m, f1_m])
model.save_weights(checkpoint_path.format(epoch=0))

print(f'Start training model')

train_history = model.fit(X_train, Y_train,
            batch_size=bs,
            epochs=epochs, #125
            validation_data=(X_val, Y_val),
            callbacks=[cp_callback],
            shuffle=True)
    
history = train_history.history
total_epochs += epochs

print(f'Training finished!')

PATH = f'./result/model_{name}/'
    
model.save(f'{PATH}_model.h5')
model.save_weights(f'{PATH}_weights.hdf5')

print(total_epochs)