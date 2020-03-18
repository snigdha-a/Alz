import argparse
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import sklearn
from keras.regularizers import l2
from keras.layers import *
from sklearn.metrics import hamming_loss,label_ranking_loss,confusion_matrix, auc,roc_curve,roc_auc_score, precision_recall_curve
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras import optimizers,losses,metrics
import keras.backend as K
from skmultilearn.utils import measure_per_label
import sklearn.metrics as skm
import tensorflow as tf
# from tensorflow_addons.metrics import HammingLoss
import scipy
from numpy import asarray
from numpy import ones
from sklearn.metrics import fbeta_score
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
globalArr= []

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predValid = self.model.predict_proba(self.validation_data[0])
        print(predValid[0,:])
        pred_binary = predValid>0.5
        Yvalid = self.validation_data[1]
        file = open("output.txt", "w")
        file.write('Epoch :'+str(batch))
        for i in range(50):
            temp = np.array_str(np.array([pred_binary[i,:],Yvalid[i,:].astype(int)]))
            file.write(temp)
            file.write('\n--------------\n')
        self.ranking_loss=label_ranking_loss(Yvalid, predValid)
        self.acc_per_label = measure_per_label(skm.accuracy_score, scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(pred_binary))
        self.f_score = measure_per_label(fbeta_non_tensor, scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(pred_binary))
        # self.prc_by_label = measure_per_label(skm.auc(skm.precision_recall_curve[0],skm.precision_recall_curve[1]), scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(predValid.round()))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(Yvalid.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(Yvalid[:, i], predValid[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        self.auc_per_label = roc_auc

        prec = dict()
        recall = dict()
        prc_auc = dict()
        for i in range(Yvalid.shape[1]):
            prec[i], recall[i], _ = precision_recall_curve(Yvalid[:, i], predValid[:, i])
            prc_auc[i] = auc(recall[i], prec[i])
        self.prc_per_label = prc_auc

        ##AUPRC for each task

        # self.val_auroc = auRoc(Yvalid.argmax(axis=1), predValid.argmax(axis=1))
        # self.val_auprc = auPrc(Yvalid.argmax(axis=1), predValid.argmax(axis=1))
        # self.val_balanced_accuracy = balancedAccuracy(Yvalid, predValid)
        # self.val_sensitivity = sensitivity(Yvalid, predValid)
        # self.val_specificity = specificity(Yvalid, predValid)
        # print("val_auroc:", self.val_auroc, ",val_auprc:", self.val_auprc, ",val_balanced_accuracy", self.val_balanced_accuracy, ",val_sensitivity:", self.val_sensitivity, ",val_specificity:", self.val_specificity)
        print("ranking Loss:",self.ranking_loss)
        # print("Per label auc:",self.auc_per_label)
        # print("Per label prc:",self.prc_per_label)
        print("Per label accuracy:",self.acc_per_label)
        globalArr.append(self.acc_per_label[1])
        print("Per label f2 score:",self.f_score)
        return

def get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length):
    model = Sequential()
    conv1_layer = Conv1D(filters=160,
                        kernel_size=8,
                        input_shape=(length, 4),
                        padding="valid",
                        activation="relu",
                        use_bias=True,kernel_regularizer=l2(0.0005),
                        bias_regularizer=l2(0.0005))
    model.add(conv1_layer)
    model.add(MaxPooling1D(pool_size=13,padding='valid'))
    model.add(Dropout(0.2))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=8,
                        filters=320,
                        use_bias=True)
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=13,strides=13))
    model.add(Dropout(0.2))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=8,
                        filters=480,
                        use_bias=True)
    model.add(convn_layer)
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(units=numLabels, use_bias=True,kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(Activation('sigmoid'))
    return model

def get_model_func(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length):
    visible = Input(shape=(1, length, 4))
    conv1_layer = Conv2D(filters=80,
                        kernel_size=(1,8),
                        input_shape=(1, length, 4),
                        padding="valid",
                        activation="relu",
                        use_bias=True)(visible)
    conv2_layer = Conv2D(filters=80,
                    kernel_size=(1,8),
                    input_shape=(1, length, 4),
                    padding="valid",
                    activation="relu",
                    use_bias=True)(visible)
    conc = concatenate([conv1_layer,conv2_layer])
    maxPool = MaxPooling2D(pool_size=(1,4),padding="valid")(conc)
    drop = Dropout(0.5)(maxPool)

    convn1_layer = Conv2D(padding="valid",
                        activation="relu",
                        kernel_size=(1,8),
                        filters=160,
                        use_bias=True)(drop)
    convn2_layer = Conv2D(padding="valid",
                        activation="relu",
                        kernel_size=(1,8),
                        filters=160,
                        use_bias=True)(drop)
    conc2 = concatenate([convn1_layer,convn2_layer])
    pool2=MaxPooling2D(pool_size=(1,4),padding="valid")(conc2)
    drop2 = Dropout(0.5)(pool2)

    con1_layer = Conv2D(padding="valid",
                        activation="relu",
                        kernel_size=(1, 8),
                        filters=240,
                        use_bias=True)(drop2)
    con2_layer = Conv2D(padding="valid",
                        activation="relu",
                        kernel_size=(1, 8),
                        filters=240,
                        use_bias=True)(drop2)
    conc3 = concatenate([con1_layer,con2_layer])
    drop3 = Dropout(0.5)(conc3)

    flat = Flatten()(drop3)
    dense = Dense(units=numLabels, use_bias=True)(flat)
    act = Activation('sigmoid')(dense)
    return Model(inputs=visible, outputs=act)

def cust_binary(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred,from_logits=True)

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    print(y_true[:, 0])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', np.unique(y_true[:, i]), y_true[:, i])
    return weights

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
    pyplot.subplot(223)
    pyplot.title('Fbeta')
    pyplot.plot(history.history['fbeta'], color='blue', label='train')
    pyplot.plot(history.history['val_fbeta'], color='orange', label='test')
    pyplot.subplot(224)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.plot(globalArr, color='red', label='mine')
    # save plot to file
    pyplot.savefig('my_plot-2.png')
    pyplot.close()

def train_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize,
                     numEpochs,
                     numConvLayers,
                     numConvFilters,
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel):

    numLabels = Y_train.shape[1]
    # X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
    # X_valid = np.reshape(X_valid, (X_valid.shape[0],1,X_valid.shape[1],X_valid.shape[2]))
    if pretrainedModel:
        model = load_model(pretrainedModel)
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length)
        optim = SGD(lr=learningRate, momentum=momentum,decay=0.001)
    class_weights = calculating_class_weights(Y_train)
    print(class_weights)
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy',fbeta])#, ranking_loss]) #specificity_metric])
    model.summary()
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1)#, save_best_only=True)#, monitor='val_specifity_metric', mode='max')
    # earlystopper = EarlyStopping(monitor='val_specificity_metric', min_delta=0, patience=60, verbose=0, mode='max')
    print(X_valid.shape)
    print(Y_valid.shape)
    print(Y_train[0])
    cust_metrics = Metrics()
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1,
    validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer,cust_metrics],class_weight={0:0,1:1,2:0,3:0,4:0,5:0,6:0,7:0})#, earlystopper])#, class_weight = classWeights)
    # learning curves
    summarize_diagnostics(history)

def fbeta(y_true, y_pred, beta=2):
	# clip predictions
    y_pred = K.clip(y_pred, 0, 1)
	# calculate elements
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=0)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=0)
    # calculate precision
    p = tp / (tp + fp + K.epsilon())
	# calculate recall
    r = tp / (tp + fn + K.epsilon())
	# calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = K.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
    return fbeta_score

def fbeta_non_tensor(y_true, y_pred, beta=2):
	# clip predictions
    y_pred = np.clip(y_pred, 0, 1)
	# calculate elements
    tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    fp = np.sum(np.round(np.clip(y_pred - y_true, 0, 1)), axis=0)
    fn = np.sum(np.round(np.clip(y_true - y_pred, 0, 1)), axis=0)
    # calculate precision
    p = tp / (tp + fp + K.epsilon())
	# calculate recall
    r = tp / (tp + fn + K.epsilon())
	# calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = np.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
    return fbeta_score

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural networnp model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=100)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=2)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.9)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=499)
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model', required=False, default=None)
    parser.add_argument('-c1w', '--class-1-weight', type=int, help='weight for positive class during training', required=False, default=1)
    parser.add_argument('-c2w', '--class-2-weight', type=int, help='weight for positive class during training', required=False, default=1)
    args = parser.parse_args()
    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)
    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)
    #no. of 1's in training dataset
    # for i in range(Y_train.shape[1]):
    #     print(np.count_nonzero(Y_train[:,i]))
    print('Model state:')
    print('LR='+str(args.learning_rate)+'\nMomentum='+str(args.momentum))
    # data = np.corrcoef(Y_train.reshape(Y_train.shape[1],Y_train.shape[0]))
    # pyplot.imshow(np.asarray(img))
    # pyplot.savefig('plt.png')
    # pyplot.close()
    # train_yhat = asarray([np.ones(Y_train.shape[1]) for _ in range(Y_train.shape[0])])
    # test_yhat = asarray([np.ones(Y_valid.shape[1]) for _ in range(Y_valid.shape[0])])
    # # print(fbeta(Y_valid,test_yhat))
    # print(measure_per_label(fbeta, scipy.sparse.csr_matrix(Y_valid),scipy.sparse.csr_matrix(test_yhat)))

    train_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
                     batchSize=args.batch_size,
                     numEpochs=args.num_epochs,
                     numConvLayers=args.num_conv_layers,
                     numConvFilters=args.num_conv_filters,
                     poolingDropout=args.pool_dropout_rate,
                     learningRate=args.learning_rate,
                     momentum=args.momentum,
                     length=args.length,
                     pretrainedModel=args.pretrained_model)
