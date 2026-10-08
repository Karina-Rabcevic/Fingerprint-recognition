import numpy as np
from PIL import Image
import tensorflow as tf

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.layers import Activation, BatchNormalization
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam


TARGET_SIZE = (128, 144)
THRESHOLD = 0.2  
WEIGHTS_PATH = 'D:/BSU/ODbKuUS/Lab_4_fings/fingers_model.weights.h5'


def read_image(filename):
    img = Image.open(filename).convert('L')
    img = img.resize(TARGET_SIZE)
    return np.array(img)


def build_base_network(input_shape):
    seq = Sequential()
    nb_filter = [6, 12]
    kernel_size = 3

    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          padding='valid', data_format="channels_first"))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size,
                          padding='valid', data_format="channels_first"))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    seq.add(Dropout(.25))

    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def load_model(weights_path):
    # Форма входа: (1, H, W) — channels_first, grayscale
    input_dim = (1, TARGET_SIZE[1], TARGET_SIZE[0])

    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_vecs_a = base_network(input_a)
    feat_vecs_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss='mse', optimizer=Adam())

    base_network.load_weights(weights_path)
    print(f'Веса загружены из {weights_path}')
    return model


def preprocess(path):
    img = read_image(path)
    img = img.astype(np.float32) / 255.0
    # Форма: (1, 1, H, W) — batch=1, channels=1
    img = img[np.newaxis, np.newaxis, :, :]
    return img


def compare(img1_path, img2_path, threshold=THRESHOLD, weights_path=WEIGHTS_PATH):
    model = load_model(weights_path)

    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)

    distance = model.predict([img1, img2], verbose=0)[0][0]

    print(f'\nРасстояние между отпечатками: {distance:.4f}')
    print(f'Порог: {threshold}')

    if distance < threshold:
        print('✅ Вывод: отпечатки СОВПАДАЮТ (один человек)')
    else:
        print('❌ Вывод: отпечатки РАЗНЫЕ (разные люди)')

    return distance


if __name__ == '__main__':
    # ↓↓↓ Укажи пути к изображениям здесь ↓↓↓
    path1 = r"D:\BSU\ODbKuUS\Lab_4_fings\data\fingers\s10\5.bmp"
    path2 = r"D:\BSU\ODbKuUS\Lab_4_fings\data\fingers\s9\4.bmp"
    # IMG1 = r'D:\BSU\ODbKuUS\Lab_4_fings\data\fingers\SOCOFing\Real\1__M_Right_thumb_finger.BMP'
    # IMG2 = r'D:\BSU\ODbKuUS\Lab_4_fings\data\fingers\SOCOFing\Real\2__F_Left_little_finger.BMP'
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    

    # compare(IMG1, IMG2, threshold=THRESHOLD, weights_path=WEIGHTS_PATH)
    compare(path1, path2, threshold=THRESHOLD, weights_path=WEIGHTS_PATH)