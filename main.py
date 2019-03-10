"""
    基于模型迁移的轴承故障诊断

    参考：
        D:\Githubs\transfer\cwru-conv1d-master
"""

import os
import sys
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

FRAME_SIZE = 400
STEP_SIZE = 20

def load_matfile(filename, frame_size=FRAME_SIZE, step_size=STEP_SIZE):
    matlab_file = scipy.io.loadmat(filename)
    DE_time = [key for key in matlab_file.keys() if key.endswith("DE_time")][0]
    FE_time = [key for key in matlab_file.keys() if key.endswith("FE_time")][0]

    signal_begin = 0
    signal_len = min(len(matlab_file[DE_time]), len(matlab_file[FE_time]))
    
    DE_samples = []
    FE_samples = []
    while signal_begin + frame_size < signal_len:
        DE_samples.append([item for sublist in matlab_file[DE_time][signal_begin:signal_begin+frame_size] for item in sublist])
        FE_samples.append([item for sublist in matlab_file[FE_time][signal_begin:signal_begin+frame_size] for item in sublist])
        signal_begin += step_size
        #signal_begin += frame_size
    
    sample_tensor = np.stack([DE_samples, FE_samples], axis=2).astype('float32')
    print("Load file {} with shape {}".format(filename, sample_tensor.shape))
    return sample_tensor

# tensor = load_matfile('data\\train\\1797.mat', frame_size=400)
# print(tensor.shape)
# print(type(tensor))

# 以文件名作为label，并予以编号
def get_labels_list(filelist):
    labels_dict = {}
    value = 0
    for filename in filelist:
        label = filename.split('.')[0]
        if not label in labels_dict:
            labels_dict[label] = value
            value += 1
    return labels_dict

# 将每个文件合并成完整的数据集
def concatenate_datasets(xd, yd, xo, yo):
    if xd is None or yd is None:
        xd = xo
        yd = yo
    else:
        xd = np.concatenate((xd, xo))
        yd = np.concatenate((yd, yo))
    return xd, yd

# 分别读取train和test目录下的文件
def read_dir(dirpath):
    dirlist = os.listdir(dirpath)
    labels_dict = get_labels_list(dirlist)
    print("labels_dict for dir {}: {}".format(dirpath, labels_dict))
    samples = None
    labels = None
    for filename in dirlist:
        sample_tensor = load_matfile(dirpath + filename)
        sample_labels = np.ones(sample_tensor.shape[0]) * labels_dict[filename.split('.')[0]]
        samples, labels = concatenate_datasets(samples, labels, sample_tensor, sample_labels)

    return samples, labels, labels_dict



# test_samples, test_labels, test_labels_dict = read_dir("data\\test\\")
# print("test features shape: {}".format(test_samples.shape))
# print("test labels shape: {}".format(test_labels.shape))

# 定义模型
import keras
import keras.backend as K
from keras import layers
from keras import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

def f1_score_macro(y_true,y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_model(input_shape, nb_classes, pre_model=None):

    """ simple version
    signal_input = Input(shape = input_shape, dtype = 'float32', name = 'input')
    x = layers.Conv1D(filters=64, kernel_size=64, activation = 'relu', name = 'conv1d_1')(signal_input)
    x = layers.MaxPooling1D(pool_size = 8, name='max_pooling1d_1')(x)
    x = layers.Flatten(name='flatten')(x)
    condition_output = layers.Dense(nb_classes, activation='softmax', name='condition')(x)
    """

    # complex version
    signal_input = Input(shape=input_shape, dtype='float32', name='input')
    conv1 = layers.Conv1D(filters=128, kernel_size=64)(signal_input)
    conv1 = layers.normalization.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.5)(conv1)

    conv2 = layers.Conv1D(filters=128, kernel_size=32)(conv1)
    conv2 = layers.normalization.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.5)(conv2)

    # conv3 = layers.Conv1D(128, kernel_size=3)(conv2)
    # conv3 = layers.normalization.BatchNormalization()(conv3)
    # conv3 = layers.Activation('relu')(conv3)
    # conv3 = layers.Dropout(0.5)(conv3)

    gap_layer = layers.pooling.GlobalAveragePooling1D()(conv2)
    condition_output = layers.Dense(nb_classes, activation='softmax', name='condition')(gap_layer)

    model = Model(signal_input, condition_output)

    if pre_model is not None:
        print("load pre_model and set_weights")
        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', 
            metrics=['categorical_crossentropy'])
    print(model.summary())

    return model


EPOCHS = 10
def train_model(x, y, pre_model=None):
    history = model.fit(x, y, epochs=EPOCHS, validation_split=0.3, 
                        callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau()])

source_data_dir = "data\\source\\"
target_data_dir = "data\\target\\"
source_model_path = "result\\source_model.hdf5"
target_model_path = "result\\target_model.hdf5"

if sys.argv[1] == "train_model":
    # 只加载source data，训练并保存结果至本地
    source_samples, source_labels, source_labels_dict = read_dir(source_data_dir)
    source_labels_cat = to_categorical(source_labels)
    print("source features shape: {}".format(source_samples.shape))
    print("source labels shape: {}".format(source_labels.shape))

    input_shape = [source_samples.shape[1], source_samples.shape[2]]
    nb_classes = len(source_labels_dict)
    model = build_model(input_shape, nb_classes, pre_model=None)

    # reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.5, patience=50,
                                                    min_lr=0.0001)
    # model checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=source_model_path, monitor='categorical_crossentropy',
                                                        save_best_only=True)
    callbacks=[reduce_lr,model_checkpoint]
    history = model.fit(source_samples, source_labels_cat, verbose=2, epochs=EPOCHS, validation_split=0.1, 
                        callbacks=callbacks)

elif sys.argv[1] == "transfer_model":
    target_samples, target_labels, target_labels_dict = read_dir(target_data_dir)
    print("target features shape: {}".format(target_samples.shape))
    print("target labels shape: {}".format(target_labels.shape))
    target_labels_cat = to_categorical(target_labels)

    x_train, x_test, y_train, y_test = train_test_split(target_samples, target_labels_cat, 
        test_size=0.2, random_state=42)

    pre_model = keras.models.load_model(source_model_path)

    input_shape = [target_samples.shape[1], target_samples.shape[2]]
    nb_classes = len(target_labels_dict)
    model = build_model(input_shape, nb_classes, pre_model)

    # reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.5, patience=50,
                                                    min_lr=0.0001)
    # model checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=target_model_path, monitor='categorical_crossentropy',
                                                        save_best_only=True)
    callbacks=[reduce_lr,model_checkpoint]
    # 怎么freeze其他层，只训练最后一层？trainable?
    history = model.fit(x_train, y_train, verbose=2, epochs=EPOCHS, validation_split=0.1, 
                        callbacks=callbacks)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    eval_result = model.evaluate(x_test, y_test)
    print(type(eval_result))
    print(eval_result)
                        
