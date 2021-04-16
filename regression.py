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
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['svg.fonttype'] = 'none'
# rcParams['font.size']=10
import matplotlib.ticker as ticker
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
import pandas as pd
import xlsxwriter
import CLR
from CLR.clr_callback import CyclicLR
from CLR import config

path = ""
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, Y, bin_sample, bin_size):
        'Initialization'
        self.Y = Y
        self.X = X
        self.n = 0
        self.bin_size = bin_size
        self.bin_sample = bin_sample
        self.bin_dict = {0:[],1:[],2:[],3:[],5:[],10:[]}
        for i in range(self.X.shape[0]):
            bin = int(self.Y[i]/10)
            if bin < 3 :
                self.bin_dict[bin].append(i)
            elif bin >=3 and bin <=4:
                self.bin_dict[3].append(i)
            elif bin >=5 and bin <=9:
                self.bin_dict[5].append(i)
            else:
                self.bin_dict[10].append(i)
        # For older data
        # self.bin_dict = {0:[],1:[],2:[],3:[],4:[]}
        # for i in range(self.X.shape[0]):
        #     if int(self.Y[i]/10) in self.bin_dict:
        #         self.bin_dict[int(self.Y[i]/10)].append(i)
        #     else:
        #         self.bin_dict[4].append(i)
        for i in self.bin_dict.keys():
            print(i, len(self.bin_dict[i]))
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
    a_mean = K.mean(x)
    b_mean = K.mean(y)
    a_norm = x - a_mean
    b_norm = y - b_mean
    numerator = K.sum(a_norm * b_norm)

    a_var = K.sum(K.square(a_norm))
    b_var = K.sum(K.square(b_norm))
    denominator = (a_var * b_var) ** 0.5

    r= numerator / denominator
    r = K.maximum(K.minimum(r, 1.0), -1.0)
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
        # bin_dict = {0:[],1:[],2:[],3:[],4:[]}
        # for i in range(Yvalid.shape[0]):
        #     if int(Yvalid[i]/10) in bin_dict:
        #         bin_dict[int(Yvalid[i]/10)].append(i)
        #     else:
        #         bin_dict[4].append(i)
        # for item,list in bin_dict.items():
        #     r,pvl = pearsonr(Yvalid.take(list),predValid.take(list))
        #     print("Pearson for bin ",item," : ",r)
        #     bin_pearson[item].append(r)
        # logs["bin4_pr"] = bin_pearson[4][-1]
        spearman,pval = spearmanr(Yvalid.flatten(),predValid.flatten())
        spearman_list.append(spearman)
        print("Validation Spearman: ",spearman)
        # predicted vs actual
        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax2= axs[0]
        ax2.title.set_text('Validation')
        ax2.plot(Y_valid, predValid,'r.',alpha=0.5)
        ax2.set_aspect('equal')
        #plot training data actual vs predicted
        pred_train = self.model.predict(self.training_data[0])
        Y_train = self.training_data[1]
        ax3=axs[1]
        ax3.title.set_text('Training')
        ax3.plot(Y_train, pred_train,'r.',alpha=0.5)
        ax3.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path+'actualvspred.png')
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
    # for i in range(len(bin_pearson)):
    #     ax3.plot(bin_pearson[i], color = colors[i], label=i)
    # ax3.plot(pearson_list, color='blue', label='test')
    ax3.plot(history.history['val_pearson_correlation'], color='blue', label='test')
    ax3.plot(history.history['pearson_correlation'], color='orange', label='train')
    ax3.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path+'loss_pearson.png')
    pyplot.close()

def get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length):
    model = Sequential()
    l1_reg = 0.00001
    l2_reg = 0.00001
    l3_reg = 0.00001
    l4_reg = 0.00001
    dropout = 0.2#0.2
    filter1 = 500
    filter2 = 250
    filter3 = 100
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
    # higher weight for negative samples
    bool_arr = K.cast(K.less(yTrue, 3),dtype='float32')
    m = K.ones_like(bool_arr)*2
    w2 = bool_arr * m
    # w = K.log(yTrue + 1) + 1
    # return K.mean(K.square(yTrue-yPred)) * (w2+1) * w
    return K.mean(K.square(yTrue-yPred))*(w2+1)

def custom_loss(yTrue,yPred):
    return 1.0-pearson_correlation(yTrue,yPred)

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
    optim = optimizers.SGD(lr=config.MIN_LR, momentum=momentum)
    clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size= config.STEP_SIZE * (X_train.shape[0] // batchSize))
    if pretrainedModel:
        model = load_model(pretrainedModel, custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
        model.compile(loss='mean_squared_error', optimizer=optim, metrics=[pearson_correlation,'mse'])
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length)
        # model.compile(loss='mean_squared_error', optimizer=optim, metrics=[pearson_correlation,'mse'])
        model.compile(loss=weighted_mse, optimizer=optim, metrics=[pearson_correlation,'mse'])


    # model.compile(loss='mean_squared_error', optimizer=optim, metrics=[pearson_correlation,'mse'])

    model.summary()

    cust_metrics = Metrics((X_train,Y_train))

    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_pearson_correlation', mode='max')
    # checkpointer = ModelCheckpoint(filepath=modelOut,
    #                                verbose=1, save_best_only=True, monitor="bin4_pr", mode='max')
    # earlystopper = EarlyStopping(monitor="bin4_pr", min_delta=0, patience=20, verbose=1, mode='max')
    earlystopper = EarlyStopping(monitor='val_pearson_correlation', min_delta=0, patience=10, verbose=0, mode='max')

    # cust_metrics = Metrics()
    #For original peaks
    # train_generator = DataGenerator(X_train,Y_train,1700,20)
    # valid_generator = DataGenerator(X_valid,Y_valid,60,60)
    # For archr peaks
    # train_generator = DataGenerator(X_train,Y_train,32000,100)
    # valid_generator = DataGenerator(X_valid,Y_valid,1500,100)
    # history = model.fit_generator(generator = train_generator, epochs=numEpochs, verbose=1,
    # validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer, cust_metrics, earlystopper])
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs,
    shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0,
    callbacks=[cust_metrics, checkpointer,  earlystopper,clr],)

    # generate actual vs pred plot for best saved model
    model = load_model(modelOut, custom_objects={'pearson_correlation':pearson_correlation,'weighted_mse': weighted_mse})
    pred_valid = model.predict(X_valid)
    fig, ax3 = plt.subplots(figsize=(3,3))
    for axis in [ax3.xaxis, ax3.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    # ax3.title.set_text('dendritic')
    ax3.plot(Y_valid, pred_valid,'r.',alpha=0.5)
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    pearson,pval = pearsonr(Y_valid.flatten(),pred_valid.flatten())
    ax3.text(1, 0, r'PR = {0:.2f}'.format(pearson),ha='right', va='bottom',
    transform=ax3.transAxes,fontsize=12)
    fig.tight_layout()
    fig.savefig(path+'actualvspred_end.svg')


    summarize_diagnostics(history)

def reverse_one_hot(sequence):
    # print(sequence.shape)
    chars = np.array(['A','G','C','T'])
    indices = np.argmax(sequence,axis=1)
    # print(indices)
    return ''.join(np.take(chars,indices))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural network model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data')
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels')
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data')
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels')
    parser.add_argument('-xT', '--xtest', help='npy file containing testing data')
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output')
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model')
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=100)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=4)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-u', '--pre-last-layer-units', type=int, help='number of sigmoid units in the layer before the output layer', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.005)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.9)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=501)
    parser.add_argument('-i', '--iter', help='model iteration', default="mono",required=False)
    parser.add_argument('-md', '--mode', help='train or test ?', required=True)
    parser.add_argument('-d', '--directory', help='path to store all other files', required=True)

    args = parser.parse_args()
    path = args.directory

    # count_dict ={}
    # for i in Y_train:
    #     key = int(i/10)
    #     if key in count_dict:
    #         count_dict[key] += 1
    #     else:
    #         count_dict[key] = 1
    # print(count_dict)
    global iter
    iter = args.iter
    mode = args.mode
    if mode=="train" :
        if (not args.xtrain or not args.ytrain or not args.xvalid or not args.yvalid or not args.model_out):
            print("Argument missing. Check xtrain, ytrain, xvalid, yvalid and model-out")
        else:
            X_train = np.load(file=args.xtrain)
            Y_train = np.load(file=args.ytrain)
            X_valid = np.load(file=args.xvalid)
            Y_valid = np.load(file=args.yvalid)

            #adding noise to Y_train and Y_valid for 0.0 labelled negative samples
            noise = np.random.normal(0.5, .1, Y_train.shape)
            y = Y_train < 1
            Y_train = y.astype(int) * noise + Y_train

            noise = np.random.normal(0.5, .1, Y_valid.shape)
            y = Y_valid < 1
            Y_valid = y.astype(int) * noise + Y_valid


            args.length = len(X_train[0])
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
                         pretrainedModel=args.pretrained_model
                        )
    elif mode=="test":
        if (not args.pretrained_model or not args.xtest or not args.yvalid):
            print("Argument missing. Check pretrained_model or xtest or yvalid")
        else:
            model = load_model(args.pretrained_model, custom_objects={'pearson_correlation':pearson_correlation})
            print("Predicting for ", args.xtest)
            X_test = np.load(file=args.xtest)
            print(X_test.shape)
            predValid = model.predict(X_test)

            # sequences=[]
            # for i in range(X_test.shape[0]):
            #     sequences.append(reverse_one_hot(X_test[i,:,:]))
            #
            # X_seq = np.stack(sequences, axis=0)
            # combined = np.hstack((X_seq.reshape(-1,1),predValid))
            # print(combined.shape)
            # # # Generate dataframe from list and write to xlsx.
            # xlsx_name= args.xtest.rsplit('/',1)[1].split(".npy")[0]
            # print("Saving in ", path+"/"+xlsx_name+".xlsx")
            # pd.DataFrame(combined).to_excel(str(path+"/"+xlsx_name+".xlsx"), engine='openpyxl',header=False, index=False)

            # Pearson correlation plot between actual and predicted for validationSet
            Y_valid = np.load(file=args.yvalid)
            fig, ax3 = plt.subplots(figsize=(3,3))
            for axis in [ax3.xaxis, ax3.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
            # ax3.title.set_text('mono')
            # ax3.plot(Y_valid, predValid,'.',alpha=0.5,color='red',label='mono Peaks')
            plt.hist(predValid,density=True,bins=50,color='red',label='mono')
            ax3.set_xlabel("Actual")
            ax3.set_ylabel("Predicted")
            plt.xticks(rotation=90)

            #comparing against another peak set
            valid_peaks = np.load(file='scate_files/cd4/differential/validationInput.npy')
            predictions = model.predict(valid_peaks)
            actual_values= np.load(file='scate_files/cd4/differential/validationLabels.npy')
            # ax3.plot(actual_values, predictions,'.',alpha=0.5,color='blue',label='cd4 Peaks')
            plt.hist(predictions,density=True,bins=50,color='blue',label='cd4')
            ax3.legend(loc="upper right")

            #comparing variances before t-test. Check if 1 value > 2 * other value.
            var1 = np.var(predValid)
            var2 = np.var(predictions)
            print(var1,var2)
            if (var1>2*var2 or var2>2*var1):
                print("Variances are NOT equal, calling Welch's t-test")
                print(ttest_ind(predValid,predictions,equal_var=False))
            else:
                print("Variances are equal, calling standard 2 sample t-test")
                print(ttest_ind(predValid,predictions,equal_var=True))

            # pearson,pval = pearsonr(Y_valid.flatten(),predValid.flatten())
            # ax3.text(1, 0, r'PR = {0:.2f}'.format(pearson),ha='right', va='bottom',
            # transform=ax3.transAxes,fontsize=12)
            fig.tight_layout()
            fig.savefig(path+"/"+'actualvspred_end.svg')
