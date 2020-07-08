import argparse
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import optimizers
from keras.regularizers import l2,l1
import keras.backend as K
from numpy import asarray
import keras.metrics
from scipy.stats import pearsonr, spearmanr
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, Y, bin_sample, bin_size):
        'Initialization'
        self.Y = Y
        self.X = X
        self.n = 0
        self.bin_size = bin_size
        self.bin_sample = bin_sample
        self.bin_dict = {0:[],1:[],2:[],3:[],4:[]}
        for i in range(self.X.shape[0]):
            if int(self.Y[i]/10) in self.bin_dict:
                self.bin_dict[int(self.Y[i]/10)].append(i)
            else:
                self.bin_dict[4].append(i)
        # for i in self.bin_dict.keys():
        #     print(i, len(self.bin_dict[i]))
        self.max = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.bin_sample/self.bin_size)

    def __getitem__(self, index):
        'Generates data containing batch_size samples'
        final_indices = []
        for item,list in self.bin_dict.items():
            final_indices += random.sample(list,self.bin_size)
        X = np.take(self.X,final_indices,axis=0)
        Y = np.take(self.Y,final_indices,axis=0)
        return X,Y

    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def pearson_correlation(x, y):
    n = K.sum(K.ones_like(x))
    sum_x = K.sum(x)
    sum_y = K.sum(y)

    sum_x_sq = K.sum(K.square(x))
    sum_y_sq = K.sum(K.square(y))

    psum = K.sum(x * y)
    num = psum - (sum_x * sum_y / n)
    den = K.sqrt((sum_x_sq - K.square(sum_x) / n) *  (sum_y_sq - K.square(sum_y) / n))

    r = num / den

    return r


# keras.metrics.pearson_correlation = pearsonr #pearson_correlation
# keras.metrics.spearman_correlation = spearmanr
from itertools import *
pearson_list = []
bin_pearson = [[0],[0],[0],[0],[0]]
spearman_list = []
class Metrics(Callback):
    def __init__(self,train_data):
        # self.validation_data = val_data
        self.training_data = train_data

    def on_epoch_end(self, batch, logs={}):
        predValid = self.model.predict(self.validation_data[0])
        Yvalid = self.validation_data[1]
        pearson,pval = pearsonr(Yvalid.flatten(),predValid.flatten())
        pearson_list.append(pearson)
        print("Validation Pearson: ",pearson)
        #bin specific pearson
        bin_dict = {0:[],1:[],2:[],3:[],4:[]}
        for i in range(Yvalid.shape[0]):
            if int(Yvalid[i]/10) in bin_dict:
                bin_dict[int(Yvalid[i]/10)].append(i)
            else:
                bin_dict[4].append(i)
        for item,list in bin_dict.items():
            r,pvl = pearsonr(Yvalid.take(list),predValid.take(list))
            print("Pearson for bin ",item," : ",r)
            bin_pearson[item].append(r)
        logs["bin4_pr"] = bin_pearson[4][-1]
        spearman,pval = spearmanr(Yvalid.flatten(),predValid.flatten())
        spearman_list.append(spearman)
        print("Validation Spearman: ",spearman)
        # predicted vs actual
        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax2= axs[0]
        ax2.title.set_text('Validation')
        ax2.plot(Y_valid, predValid,'r.',alpha=0.5)
        # hb = ax2.hexbin(Y_valid.flatten(), predValid.flatten(), gridsize=10, cmap='inferno', alpha=0.5,bins='log')
        # cb = fig.colorbar(hb, ax=ax2)
        # cb.set_label('log10(N)')

        # plt.savefig("output.png")
        #plot training data actual vs predicted
        pred_train = self.model.predict(self.training_data[0])
        Y_train = self.training_data[1]
        ax3=axs[1]
        ax3.title.set_text('Training')
        ax3.plot(Y_train, pred_train,'r.',alpha=0.5)
        # plt.savefig("output_train.png")
        fig.tight_layout()
        fig.savefig('output.png')
        pyplot.close()
        print("Hello")


def summarize_diagnostics(history):
    gs = gridspec.GridSpec(2,1)
    fig = pyplot.figure()

    ax2=pyplot.subplot(gs[0, :])
    ax2.title.set_text('Loss')
    ax2.plot(history.history['val_loss'], color='blue', label='test')
    ax2.plot(history.history['loss'], color='orange', label='train')
    ax2.legend(loc="lower right")

    ax3=pyplot.subplot(gs[1, :])
    ax3.title.set_text('Pearson')
    colors = ['green','pink','red','cyan','magenta']
    for i in range(len(bin_pearson)):
        ax3.plot(bin_pearson[i], color = colors[i], label=i)
    # ax3.plot(pearson_list, color='blue', label='test')
    ax3.plot(history.history['val_pearson_correlation'], color='blue', label='test')
    ax3.plot(history.history['pearson_correlation'], color='orange', label='train')
    ax3.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig('reg_plot.png')
    pyplot.close()

def get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length):
    model = Sequential()
    l1_reg = 0.00
    l2_reg = 0.00
    l3_reg = 0.001
    l4_reg = 0.003
    dropout = 0.20
    filter1 = 1000
    filter2 = 500
    filter3 = 250
    conv1_layer = Conv1D(filters=filter1,
                        kernel_size=8,
                        input_shape=(length, 4),
                        padding="valid",
                        activation="relu",
                        use_bias=True, kernel_regularizer=l2(l1_reg))
                        # use_bias=True)
    model.add(conv1_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=4,
                        filters=filter2,
                        use_bias=True, kernel_regularizer=l2(l2_reg))
                        # use_bias=True)
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=4,
                        filters=filter3,
                        use_bias=True, kernel_regularizer=l2(l3_reg))
                        # use_bias=True)
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(units=numLabels, activation='relu', use_bias=True, kernel_regularizer=l2(l4_reg)))
    print("Regularization values: ",l1_reg,l2_reg,l3_reg,l4_reg)
    print("Dropout values: ",dropout)
    print("Filter values: ", filter1,filter2,filter3)
    return model

def weighted_mse(yTrue,yPred):
    bool_arr = K.cast(K.greater(yTrue/10, 3),dtype='float32')
    m = K.ones_like(bool_arr)*4
    w2 = bool_arr * m
    w = K.log(yTrue + 1) + 1
    return K.mean(K.square(yTrue-yPred)) * (w2+1) * w


def train_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize,
                     numEpochs,
                     numConvLayers,
                     numConvFilters,
                     preLastLayerUnits,
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel,
                    ):
    numLabels = 1
    if pretrainedModel:
        model = load_model(pretrainedModel, custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[pearson_correlation,'mse'])
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length)
        model.compile(loss=weighted_mse, optimizer='adam', metrics=[pearson_correlation,'mse'])

    optim = optimizers.SGD(lr=learningRate, momentum=momentum)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=[pearson_correlation,'mse'])

    model.summary()

    cust_metrics = Metrics((X_train,Y_train))

    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_pearson_correlation', mode='max')
    # checkpointer = ModelCheckpoint(filepath=modelOut,
    #                                verbose=1, save_best_only=True, monitor="bin4_pr", mode='max')
    # earlystopper = EarlyStopping(monitor="bin4_pr", min_delta=0, patience=20, verbose=1, mode='max')
    earlystopper = EarlyStopping(monitor='val_pearson_correlation', min_delta=0, patience=20, verbose=0, mode='max')

    # cust_metrics = Metrics()

    train_generator = DataGenerator(X_train,Y_train,1700,20)
    valid_generator = DataGenerator(X_valid,Y_valid,60,60)
    # history = model.fit_generator(generator = train_generator, epochs=numEpochs, verbose=1,
    # validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer, cust_metrics, earlystopper])
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1,
    validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[cust_metrics, checkpointer,  earlystopper])
    summarize_diagnostics(history)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural network model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=250)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=4)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-u', '--pre-last-layer-units', type=int, help='number of sigmoid units in the layer before the output layer', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.9)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=501)
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model', required=False, default="best.hdf5")

    args = parser.parse_args()

    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)
    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)

    # model1 = load_model("signal_all.hdf5", custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
    # model2 = load_model("best.hdf5", custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
    # model3 = load_model("output-oldModel.hdf5", custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
    # model4 = load_model("output-run1.hdf5", custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
    # pr = {1:[],2:[],3:[],4:[]}
    # for i in range(20):
    #     final_indices = []
    #     for j in range(Y_valid.shape[0]):
    #         if int(Y_valid[j]) > i:
    #             final_indices.append(j)
    #     X = np.take(X_valid,final_indices,axis=0)
    #     Y = np.take(Y_valid,final_indices,axis=0)
    #     for i,m in enumerate([model1, model2,model3,model4]):
    #         predValid = m.predict(X)
    #         pearson,pval = pearsonr(Y,predValid.flatten())
    #         pr[i+1].append(pearson)
    # colors = ['green','pink','red','cyan','magenta']
    # for k in pr.keys():
    #     plt.plot([i for i in range(20)], pr[k],colors[k-1],label=k)
    # plt.savefig("cutoff.png")

    print(X_train.shape)
    print(X_valid.shape)


    train_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
                     batchSize=args.batch_size,
                     numEpochs=args.num_epochs,
                     numConvLayers=args.num_conv_layers,
                     numConvFilters=args.num_conv_filters,
                     preLastLayerUnits=args.pre_last_layer_units,
                     poolingDropout=args.pool_dropout_rate,
                     learningRate=args.learning_rate,
                     momentum=args.momentum,
                     length=args.length,
                     pretrainedModel=args.pretrained_model,
                    )
