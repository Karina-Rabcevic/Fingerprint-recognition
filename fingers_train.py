import glob
import random
import numpy as np
from PIL import Image, ImageFile, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Activation, BatchNormalization
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

TARGET_SIZE = (128, 144)  


def read_image(filename, augment_img=False):
    img = Image.open(filename).convert('L')
    img = img.resize(TARGET_SIZE)

    if augment_img:
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

    return np.array(img)


def get_subject_files(subject_id):
    return sorted(glob.glob(f'data/fingers/s{subject_id}/*.bmp'))


size = 1  
total_sample_size = 4000
num_subjects = 10


def get_data(size, total_sample_size, augment_img=False):
    subject_files = {i: get_subject_files(i) for i in range(1, num_subjects + 1)}

    for i, files in subject_files.items():
        if len(files) < 2:
            print(f'Внимание: у субъекта s{i} только {len(files)} файлов')

    sample_file = subject_files[1][0]
    image = read_image(sample_file)
    image = image[::size, ::size]
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0

    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_genuine = np.zeros([total_sample_size, 1])

    samples_per_subject = total_sample_size // num_subjects

    for i in range(1, num_subjects + 1):
        files = subject_files[i]
        n = len(files)
        for _ in range(samples_per_subject):
            ind1, ind2 = 0, 0
            while ind1 == ind2:
                ind1 = np.random.randint(n)
                ind2 = np.random.randint(n)

            img1 = read_image(files[ind1], augment_img)
            img2 = read_image(files[ind2], augment_img)

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            y_genuine[count] = 1
            count += 1

    x_geuine_pair = x_geuine_pair[:count]
    y_genuine = y_genuine[:count]

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    valid_subjects = [i for i in range(1, num_subjects + 1) if len(subject_files[i]) >= 1]

    for i in range(total_sample_size // num_subjects):
        for _ in range(num_subjects):

            while True:
                s1 = np.random.choice(valid_subjects)
                s2 = np.random.choice(valid_subjects)
                if s1 != s2:
                    break

            files1 = subject_files[s1]
            files2 = subject_files[s2]

            img_idx1 = np.random.randint(len(files1))
            img_idx2 = np.random.randint(len(files2))

            img1 = read_image(files1[img_idx1], augment_img)
            img2 = read_image(files2[img_idx2], augment_img)

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2

            y_imposite[count] = 0
            count += 1

    x_imposite_pair = x_imposite_pair[:count]
    y_imposite = y_imposite[:count]

    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y


X, Y = get_data(size, total_sample_size, augment_img=True)

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)


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


input_dim = x_train.shape[2:]
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(input_a)
feat_vecs_b = base_network(input_b)


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])


epochs = 200
adam = Adam()

model = Model(inputs=[input_a, input_b], outputs=distance)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))


model.compile(loss=contrastive_loss, optimizer=adam)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1]

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2,
          epochs=epochs, callbacks=[early_stop])

pred = model.predict([x_test[:, 0], x_test[:, 1]])

# ROC-AUC
scores = -pred.ravel()
auc = roc_auc_score(y_test.ravel(), scores)
print(f'ROC-AUC: {auc:.4f}')

fpr, tpr, _ = roc_curve(y_test.ravel(), scores)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Fingerprints')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training history')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()


thresholds = np.arange(0.1, 1.5, 0.05)
best_acc, best_t = 0, 0.5
for t in thresholds:
    acc = accuracy_score(y_test.ravel(), (pred.ravel() < t).astype(int))
    if acc > best_acc:
        best_acc, best_t = acc, t

print(f'Лучший порог: {best_t:.2f}')
print(f'Точность при этом пороге: {best_acc}')

base_network.save_weights('fingers_model.weights.h5')
print('Веса сохранены в fingers_model.weights.h5')