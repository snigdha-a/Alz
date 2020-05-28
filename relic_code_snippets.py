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
