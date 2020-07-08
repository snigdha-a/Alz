class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        try:
            self.ranking_loss=label_ranking_loss(Yvalid, predValid)
            print("ranking Loss:",self.ranking_loss)
        except:
            print("BINARY situation")

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', np.unique(y_true[:, i]), y_true[:, i])
    # return weights
    return compute_sample_weight([{0:0,1:0},{0:1,1:1},{0:0,1:0},{0:0,1:0},{0:0,1:0},{0:0,1:0},{0:0,1:0},{0:0,1:0}], y_true)
    # return weights

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets. pos_weight > 1 decreases no. of
    false negatives. < 1 decreases False Positives.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    unweighted_losses = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=10)
    weighted_losses = unweighted_losses * global_weights
    return tf.reduce_mean(loss, axis=-1)

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

# def fbeta(y_true, y_pred, beta=2):
# 	# clip predictions
#     y_pred = K.clip(y_pred, 0, 1)
# 	# calculate elements
#     tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
#     fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=0)
#     fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=0)
#     # calculate precision
#     p = tp / (tp + fp + K.epsilon())
# 	# calculate recall
#     r = tp / (tp + fn + K.epsilon())
# 	# calculate fbeta, averaged across each class
#     bb = beta ** 2
#     fbeta_score = K.mean((1 + bb) * (p * r) / (bb * p + r + K.epsilon()))
#     return fbeta_score
def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2
    # y_pred = y_pred>0.5
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

#signal_extraction.py
def generateTrainingData():
    # converts peaks and labels to npy format
    path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
    for cluster in range(1,9):
        with open(path+str(cluster)+"_hg19.bed") as f1,
        open('signal_data/'+str(cluster)+'label') as f2:
            for peak, label in zip(f1,f2):
                peak = line.strip().split()
                chromosome = peak[0]
                item = (chromosome,peak[1],peak[2])
                if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
                    testSet.append((item,label[0][0]))
                elif chromosome.startswith('chr4'):
                    validationSet.append((item,label[0][0]))
                else:
                    trainingSet.append((item,label[0][0]))
    generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
    generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
    generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')

def generateSignals():
    # load peaks.txt into a list
    peaks=[]
    open_file = open(path+'GSE129785_scATAC-Hematopoiesis-All.peaks.txt')
    for line in open_file:
        peaks.append(line.rstrip('\n'))
    print("Done with peaks list")
    path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
    with open(path+str(cluster)+"_hg19.bed") as f:
        peak = re.sub('  ','_',line.rstrip('\n'))


#regression.py
def get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length):
    input_list = []
    cnn_list = []
    from keras.layers import Input
    for i in range(length-4):
        input = Input(shape=(4, 4))
        cnn = Conv1D(kernel_size=2,
                            filters=250)(input)
                            # use_bias=True)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn_list.append(cnn)
        input_list.append(input)
    from keras.models import Model
    from keras.layers.merge import concatenate
    merge = concatenate(cnn_list)
    output = Dense(units=numLabels, use_bias=True)(merge)
    model = Model(inputs=input_list, outputs=output)
    return model

    X_train_list = []
    for i in range(X_train.shape[1]-4):
        X_train_list.append(X_train[:,i:i+4,:])
    X_valid_list = []
    for i in range(X_valid.shape[1]-4):
        X_valid_list.append(X_valid[:,i:i+4,:])

    ax1=pyplot.subplot(gs[0, :])
    ax1.title.set_text('All metrics')
    ax1.plot(history.history['val_mse'], color='blue', label='mse')
    ax1.plot(history.history['val_mae'], color='orange', label='mae')
    ax1.plot(history.history['val_cosine'], color='cyan', label='cosine')
    ax1.plot(history.history['val_msle'], color='black', label='msle')
    ax1.plot(spearman_list,color='pink',label='spearman')
    ax1.legend(loc="lower right")

    predValid = self.model.predict(self.validation_data[0:497])
    Yvalid = self.validation_data[497]

        # scalery = PowerTransformer()
        # Yvalid = scalery.inverse_transform(Yvalid)
        # predValid = scalery.inverse_transform(predValid)
        # Yvalid = np.exp(Yvalid)
        # predValid = np.exp(predValid)

    #checking hexbin plots
    print("Loading model")
    model = load_model("best.hdf5", custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
    print("Predicting")
    predValid = model.predict(X_valid)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax2 = axs[0]
    ax2.title.set_text('Validation')
    hb = ax2.hexbin(Y_valid.flatten(), predValid.flatten(), gridsize=10, cmap='inferno', alpha=0.5,bins='log')
    ax2.axis([Y_valid.min(), Y_valid.max(), predValid.min(), predValid.max()])
    cb = fig.colorbar(hb, ax=ax2)

    ax3 = axs[1]
    ax3.title.set_text('Training')
    predtrain = model.predict(X_train)
    hb = ax3.hexbin(Y_train.flatten(), predtrain.flatten(), gridsize=10, cmap='inferno', alpha=0.5,bins='log')
    ax3.axis([Y_train.min(), Y_train.max(), predtrain.min(), predtrain.max()])
    cb = fig.colorbar(hb, ax=ax3)

    fig.tight_layout()
    fig.savefig('output.png')
    pyplot.close()

    # Y_train = Y_train.reshape(-1, 1)
    # Y_valid = Y_valid.reshape(-1, 1)
    # scalery = PowerTransformer()
    # Y_train = scalery.fit_transform(Y_train)
    # Y_valid = scalery.fit_transform(Y_valid)
    # Y_valid = np.log1p(Y_valid)
    # Y_train = np.log1p(Y_train)

    # test_yhat = np.random.randint(0,120,Y_train.shape[0])
    # print(type(test_yhat))
    # print(type(Y_train))
    # print(pearsonr(Y_train.flatten(),test_yhat))
    # #
    # test_yhat = np.random.randint(0,120,Y_valid.shape[0])
    # print(type(test_yhat))
    # print(type(Y_valid))
    # print(pearsonr(Y_valid.flatten(),test_yhat))

        #generator version
        # for batch_index in range(3):
        #     xVal, Yvalid = next(self.validation_data)
        #     predValid = self.model.predict(xVal)
        #     pearson,pval = pearsonr(Yvalid.flatten(),predValid.flatten())
        #     pearson_list.append(pearson)
        #     print("Validation Pearson: ",pearson)
        #     spearman,pval = spearmanr(Yvalid.flatten(),predValid.flatten())
        #     spearman_list.append(spearman)
        #     print("Validation Spearman: ",spearman)
        #     # predicted vs actual
        #     plt.plot(Yvalid, predValid, 'ro')
        #     plt.savefig("output.png")
