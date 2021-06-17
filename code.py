import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv3D, MaxPooling2D
from keras.layers.core import Flatten
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from numpy import array
import time
import seaborn as sns
from matplotlib import pyplot
import os
import shutil
from keras.layers import ConvLSTM2D
import joblib
from keras.layers import LeakyReLU, Dropout
from keras import layers
import math
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from adabelief_tf import AdaBeliefOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score
import seaborn as sns
import itertools
from keras.initializers import RandomNormal
from sklearn import svm

np.set_printoptions(suppress=True)
colormap = pyplot.cm.coolwarm_r
location = 'Taipei101'
lon, lat = 121.565, 25.035  # Taipei101
latent_dim = 100
batchsize = 30  # 30
epoch = 60  # 60 30
look_back = 3
long = 1  # 前幾小時    1~6
version = 10  # 10
len_pre = 8760 - look_back - long + 1


def discriminator_model():
    model = Sequential()
    model.add(
        Conv2D(32, (2, 2),  # 16
               padding='same',
               input_shape=(3, 3, 1)))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    print(g.summary())
    print(d.summary())
    d.trainable = False
    model.add(d)
    return model


def combine_images(image):
    return image.mean(axis=0)


def P_train(BATCH_SIZE, v, n):
    X_train = np.load(f'./{location}/save_z_{n}sig.npy')  # float64
    temp = X_train.reshape(-1, 1).tolist()
    X_train_max = max(map(max, temp))  # 23830.0 >>  11,915
    X_train_max_half = X_train_max / 2
    X_train = (X_train.astype(np.float32) - X_train_max_half) / X_train_max_half
    X_train = X_train[:, :, :, None]  # (8760, 3, 3, 1)

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    g.compile(loss='mse', optimizer='adam')  # adam
    g_d_optim = RMSprop(lr=0.001, clipvalue=1.0, decay=1e-8)
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_d_optim)  # binary_crossentropy
    d.trainable = True
    d_optim = RMSprop(lr=0.001, clipvalue=1.0, decay=1e-8)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)  # binary_crossentropy

    losses = []
    rmse = []

    for ee in range(epoch):  # 30  100
        # print("Epoch is　：", epoch)

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, latent_dim))
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # real
            generated_images = g.predict(noise, verbose=0)  # fake
            X = np.concatenate((image_batch, generated_images), axis=0)
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            y = np.asarray(y).astype('float32').reshape((-1, 1))

            d_loss = d.train_on_batch(X, y)
            # G固定
            # g_loss = d_on_g.train_on_batch(noise, np.asarray([1] * BATCH_SIZE).astype('float32').reshape((-1, 1)))

            # print("batch %d  d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, latent_dim))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.asarray([1] * BATCH_SIZE).astype('float32').reshape((-1, 1)))
            d.trainable = True
            # print("batch %d  g_loss : %f" % (index, g_loss))

            if index == int(X_train.shape[0] / BATCH_SIZE) - 1:
                fake = combine_images(generated_images)
                fake = fake * X_train_max_half + X_train_max_half
                real = combine_images(image_batch)  # 平均
                real = real * X_train_max_half + X_train_max_half

                np.save(f'./{location}/{n}sig/train_save_png_npy/P/fake/ver{str(v)}_epoch{str(ee)}.npy', fake)
                np.save(f'./{location}/{n}sig/train_save_png_npy/P/real/ver{str(v)}_epoch{str(ee)}.npy', real)

                fake = fake.reshape(3, 3)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle("Fake vs Real_epoch:" + str(ee))

                sns.heatmap(ax=ax[0], data=fake, linewidths=0.5, vmin=500, vmax=10000, cmap=colormap)
                ax[0].set_title('Fake')

                real = real.reshape(3, 3)
                sns.heatmap(ax=ax[1], data=real, linewidths=0.5, vmin=500, vmax=10000, cmap=colormap)
                ax[1].set_title('Real')

                fig.savefig(
                    f'./{location}/{n}sig/train_save_png_npy/P/fake_vs_real/ver{str(v)}_epoch{str(ee)}.png')
                # fig.show()

                fake = fake.reshape(3, 3, 1)
                real = real.reshape(3, 3, 1)

                ###########################################################################
                g.save_weights(f'./{location}/{n}sig/model/g_P_v{str(v)}.h5')
                d.save_weights(f'./{location}/{n}sig/model/d_P_v{str(v)}.h5')
                fake = fake.reshape(1, 3, 3, 1)

                R_F_rmse = np.sqrt(mean_squared_error(real.reshape(-1, 1), fake.reshape(-1, 1)))
                # print(R_F_rmse)
                losses.append((d_loss, g_loss))
                rmse.append(R_F_rmse)

    return losses, rmse


def GAN_plot(losses, rmse, v, ty, n):
    fig, ax = plt.subplots(2, figsize=(20, 7))
    fig.suptitle("Training Losses  epoch:" + str(epoch) + 'bath size :' + str(batchsize))
    losses = np.array(losses)
    ax[0].plot(losses.T[0], label='Discriminator  Loss')
    ax[0].plot(losses.T[1], label='Generator  Loss')
    ax[0].legend()
    ax[1].plot(rmse, label='Rmse')
    ax[1].legend()
    print(f'v_{str(v)}', np.min(rmse), np.max(rmse), np.mean(rmse))
    fig.savefig(f'./{location}/{n}sig/train_save_png_npy/{ty}_v{str(v)}.png')

    fig.show()
    return rmse[-1]


def PN_GAN_train(n):
    start = time.time()
    a = []

    for v in range(version):
        print(f'version:{str(v)}   type:P')
        losses, rmse = P_train(batchsize, v, n)
        rmse_last = GAN_plot(losses, rmse, v, 'P', n)
        a.append(rmse_last)
    end = time.time()
    print(end - start)
    print('rmse_last:', a)


def PN_GAN_predict(lon, lat, n):
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)
    center = f'{lon}_{lat}'
    gr_test = pd.read_csv(f'./grid_data_test/{center}.csv')
    data_2018 = np.load(f'./{location}/save_z_train.npy')
    data_2019 = np.load(f'./{location}/save_z_test.npy')

    temp = data_2018.reshape(-1, 1).tolist()
    train_max = max(map(max, temp))
    train_max_half = train_max / 2
    data_2018_v2 = (data_2018.astype(np.float32) - train_max_half) / train_max_half
    data_2019_v2 = (data_2019.astype(np.float32) - train_max_half) / train_max_half
    train = data_2018_v2[:, :, :, None]  # (8760, 3, 3, 1)
    test = data_2019_v2[:, :, :, None]  # (8760, 3, 3, 1)
    result = pd.DataFrame()

    # 產生集成前的data
    for year in ['train', 'test']:
        print(year)
        data = locals()[year]
        for v in list(range(version)):
            d_P = discriminator_model()
            d_P.load_weights(f'./{location}/{n}sig/model/d_P_v{str(v)}.h5')
            result[f'P_v{str(v)}'] = list(d_P.predict(data, verbose=0).reshape(-1))

        result['TIME'] = gr_test['TIME']
        result['POPULATION'] = gr_test['POPULATION']
        result.to_csv(f'./{location}/result_{year}_{n}sig.csv', index=False, encoding='utf-8-sig')
        # 有了result檔 每個column是每個模型的預測結果


def Distance(x, y):
    # print(y-x)
    # return sum(sum((y - x)))[0]
    return np.min(y - x)


def anomaly_score(lon, lat, n, BATCH_SIZE):
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)
    center = f'{lon}_{lat}'
    gr_test = pd.read_csv(f'./grid_data_test/{center}.csv')
    data_2018 = np.load(f'./{location}/save_z_train.npy')
    data_2019 = np.load(f'./{location}/save_z_test.npy')
    temp = data_2018.reshape(-1, 1).tolist()
    train_max = max(map(max, temp))
    train_max_half = train_max / 2
    train = data_2018[:, :, :, None]
    test = data_2019[:, :, :, None]

    result_score = pd.DataFrame()
    for year in ['train', 'test']:
        print(year)
        data = locals()[year]
        for v in list(range(version)):
            g_P = generator_model()
            g_P.load_weights(f'./{location}/{n}sig/model/g_P_v{str(v)}.h5')

            noise = np.random.uniform(-1, 1, size=(8760, latent_dim))
            # g pre 出來要反正規化 --> data 要沒正規化過的
            g_P_predict = g_P.predict(noise, verbose=0) * train_max_half + train_max_half

            g_P_anomaly_score, g_N_anomaly_score = [], []
            for k in range(8760):
                P_score = Distance(data[k], g_P_predict[k])
                g_P_anomaly_score.append(P_score)
            result_score[f'P_v{str(v)}'] = g_P_anomaly_score

        result_score['TIME'] = gr_test['TIME']
        result_score['POPULATION'] = gr_test['POPULATION']
        result_score.to_csv(f'./{location}/result_score_{year}_{n}sig.csv', index=False, encoding='utf-8-sig')


def second_model(lon, lat, n):
    delta = 0.005
    lon = lon
    lat = lat
    right = str('%.03f' % (lon + delta))
    left = str('%.03f' % (lon - delta))
    top = str('%.03f' % (lat + delta))
    down = str('%.03f' % (lat - delta))
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)
    print(right, left, top, down)
    left_top, top, right_top = f'{left}_{top}', f'{lon}_{top}', f'{right}_{top}'
    left_center, center, right_center = f'{left}_{lat}', f'{lon}_{lat}', f'{right}_{lat}'
    left_down, down, right_down = f'{left}_{down}', f'{lon}_{down}', f'{right}_{down}'

    train_threshold = pd.read_csv(f'./{location}/{n}sig/threshold_{str(center)}.csv')
    gr_test1 = pd.read_csv(f'./grid_data_test/{left_top}.csv')
    gr_test2 = pd.read_csv(f'./grid_data_test/{top}.csv')
    gr_test3 = pd.read_csv(f'./grid_data_test/{right_top}.csv')
    gr_test4 = pd.read_csv(f'./grid_data_test/{left_center}.csv')
    gr_test5 = pd.read_csv(f'./grid_data_test/{center}.csv')
    gr_test6 = pd.read_csv(f'./grid_data_test/{right_center}.csv')
    gr_test7 = pd.read_csv(f'./grid_data_test/{left_down}.csv')
    gr_test8 = pd.read_csv(f'./grid_data_test/{down}.csv')
    gr_test9 = pd.read_csv(f'./grid_data_test/{right_down}.csv')

    train_X = pd.read_csv(f'./{location}/result_train_{n}sig.csv').iloc[:, :-2]
    test_X = pd.read_csv(f'./{location}/result_test_{n}sig.csv').iloc[:, :-2]
    print(train_X)
    Emsem_result = pd.DataFrame()
    train_y = pd.read_csv(f'./{location}/train_target/{n}sig/all.csv')['Sum']
    test_y = pd.read_csv(f'./{location}/test_target/{n}sig/all.csv')['Sum']

    model = RandomForestClassifier(n_estimators=500)
    model.fit(train_X, train_y)
    joblib.dump(model, f"./{location}/{n}sig/emsemble_model.joblib")  # save
    y_pred = model.predict(test_X)
    print('-------sigma-------:', n)
    print(metrics.classification_report(test_y, y_pred))
    print('f1_score:', f1_score(test_y, y_pred))
    print('recall', recall_score(test_y, y_pred))

    Emsem_result['TIME'] = gr_test1['TIME']
    Emsem_result['left_top_POP'] = gr_test1['POPULATION']
    Emsem_result['top_POP'] = gr_test2['POPULATION']
    Emsem_result['right_top_POP'] = gr_test3['POPULATION']
    Emsem_result['left_POP'] = gr_test4['POPULATION']
    Emsem_result['center_POP'] = gr_test5['POPULATION']
    Emsem_result['right_POP'] = gr_test6['POPULATION']
    Emsem_result['left_down_POP'] = gr_test7['POPULATION']
    Emsem_result['down_POP'] = gr_test8['POPULATION']
    Emsem_result['right_down_POP'] = gr_test9['POPULATION']
    Emsem_result['predict_y'] = y_pred
    Emsem_result['test_y'] = test_y
    Emsem_result['location'] = train_threshold['location']
    Emsem_result['threshold'] = train_threshold['threshold']
    col_list = list(Emsem_result.columns)
    for i in range(9):
        th = Emsem_result.at[i, 'threshold']
        loca = Emsem_result.at[i, 'location']
        Emsem_result[f'{loca}_target'] = Emsem_result[col_list[i + 1]].apply(lambda x: 0 if x <= th else 1)
    Emsem_result.to_csv(f'./{location}/{n}sig/result_emsemble.csv', index=False, encoding='utf-8-sig')


def anomaly_model(lon, lat, n):
    PN_GAN_train(n)
    PN_GAN_predict(lon, lat, n)
    anomaly_score(lon, lat, n, batchsize)
    second_model(lon, lat, n)
