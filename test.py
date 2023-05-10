import random
import numpy as np

from sklearn.metrics import classification_report
from pathlib import Path
from keras.models import load_model
from PIL import Image, ImageChops, ImageFilter
from keras import backend as K

img_path = Path(r'Path-to-dataset-folder')
MAX_SIZE = (270, 115)
MIN_SIZE = (90, 40)
MEAN_SIZE = (int(np.mean([MIN_SIZE[0], MAX_SIZE[0]])), int(np.mean([MIN_SIZE[1], MAX_SIZE[1]])))

words_list_ru = ['и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'этот', 'к', 'но', 'они', 'мы', 'тут', 'из', 'у', 'который', 'то', 'за', 'свой',
 'сейчас', 'весь', 'год', 'от', 'так', 'о', 'для', 'ты', 'же', 'все', 'тот', 'мочь', 'вы', 'человек', 'такой', 'его', 'сказать', 'только', 'или', 'ещё', 'бы', 'себя', 'один', 'как', 'уже',
 'до', 'время', 'если', 'сам', 'когда', 'другой', 'вот', 'говорить', 'наш', 'мой', 'знать', 'стать', 'при', 'чтобы', 'дело', 'жизнь', 'кто', 'первый', 'очень', 'два', 'день', 'её', 'новый', 'рука', 'даже',
 'во', 'со', 'раз', 'где', 'там', 'под', 'можно', 'ну', 'какой', 'после', 'их', 'работа', 'без', 'самый', 'потом', 'надо', 'хотеть', 'ли', 'слово', 'идти', 'большой', 'должен', 'место', 'иметь', 'ничто']


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


def get_data_test(classes_count, include_augmentation=True):
    # RGB
    img_path_test = img_path / 'test'
    X_test = []
    y_test = []
    imgs = []
    augmentation_ways = ['original',] #  'shift', 'rotate',  'blur'
    
    if include_augmentation:
        augmentation_ways = ['original', 'blur', 'rotate']
    
    
    class_dot = 0
    for word in words_list_ru[:classes_count]:
        index = 0
        print(word, class_dot)

        for word_path in img_path_test.glob(f'{word}_*'):
            img = Image.open(word_path)

            for way in augmentation_ways:

                res_img = augmentation(img, way)
                imgs.append(res_img)
                
                res_img = res_img.resize(MIN_SIZE)    
                img_arr = np.array(res_img)

                
                X_test.append(img_arr)
                y_test.append([class_dot])

            index += 1
        class_dot += 1


    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')

    # Normaliz
    X_test = X_test / 255
    
    return  X_test, y_test, imgs


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

models_PATH = Path(r'F:\Dev\Jupyter projects\git_projects\WordNet\full_models')

vgg16 = models_PATH / 'VGG16'
vgg19 = models_PATH / 'VGG19'
resnet50 = models_PATH / 'ResNet50'
resnet101 = models_PATH / 'ResNet101'
my_model = models_PATH / 'MyNet'

picked_model = resnet101
 
model_path = list(picked_model.glob('*.h5'))[0]
weights_path = list(picked_model.glob('*.hdf5'))[0]
# my_weights_path = list(picked_model.glob('checkpoint'))[0]
model = load_model(model_path, custom_objects={'recall_m':recall_m, 'precision_m':precision_m, 'f1_m':f1_m})
model.load_weights(weights_path)

X_test, y_test, imgs = get_data_test(100)

bs=8
classes_count = 100
y_pred = model.predict(X_test, batch_size=bs, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_bool, output_dict=True)

sum_prec = 0
sum_recall = 0
sum_f1 = 0

for key in map(str, range(classes_count)):
    sum_prec += report[key]['precision']
    sum_recall += report[key]['recall']
    sum_f1 += report[key]['f1-score']

mean_acc = report['accuracy']
mean_prec = sum_prec / classes_count
mean_recall = sum_recall / classes_count
mean_f1 = sum_f1 / classes_count

print(classification_report(y_test, y_pred_bool))
print(f'accuracy - {mean_acc}')
print(f'precision - {mean_prec}')
print(f'recall - {mean_recall}')
print(f'f1 - {mean_f1}')




