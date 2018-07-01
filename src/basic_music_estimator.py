import tensorflow as tf
import pandas as pd
import numpy as np

NUM_LAYERS = 1
NUM_NEURONS = 10
LEARNING_RATE = 0.01
SAVE_PATH = "../saves/"
FILE_TRAIN = "../data/four_schuberts_16ts.pkl"

# only one feature column right now
NUM_TIMESTEPS = 16
NUM_KEYS = 128

feature_columns = [tf.feature_column.numeric_column("x", shape=(NUM_TIMESTEPS, NUM_KEYS))]



def my_input_fn(df, repeat_count=1, shuffle_count=1):

    # then return the batch_features and batch_labels
    # dictionary's keys are the names of the features, and the dictionary's values are the feature's values
    batch_features = {'x': [tf.pack(np.zeros((NUM_TIMESTEPS, NUM_KEYS)))]}

    # which is a list of the label's values for a batch
    batch_labels = [np.zeros((NUM_TIMESTEPS, NUM_KEYS))]

    print(batch_features)
    print(batch_labels)

    return batch_features, batch_labels




def my_model_fn(
        features,   # =batch_features
        labels,     # =batch_labels
        mode        # instance of tf.estimator.ModeKeys
):
    """
    Function must:
        -create architecture of model
        -define behavior in TRAIN, EVAL, PREDICT modes
    """

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    print(input_layer)
    # creating architecture
    # X_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, VOCAB_SIZE])

    # need to take labels and create the y_placeholder equivalent
    # y_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, VOCAB_SIZE])


    def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS, activation=tf.nn.relu)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(NUM_LAYERS)])

    lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(stacked_lstm, output_size=NUM_KEYS)

    outputs, states = tf.nn.dynamic_rnn(lstm_with_wrapper, input_layer, dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train = optimizer.minimize(loss)

    # sentence_loss_pl = tf.placeholder(tf.float32, [])

    # init = tf.global_variables_initializer()

    # my_checkpoint_config = tf.estimator.RunConfig(keep_checkpoint_max=2)

    # return init, train, loss, X_placeholder, y_placeholder, outputs, sentence_loss_pl

    loss_summary = tf.summary.scalar('Loss', loss)

    predictions = outputs

    # if using the model to predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        print("predicting")
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # if evaluating the model
    elif mode == tf.estimator.ModeKeys.EVAL:
        print("evaluating")
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss': loss_summary})

    # training the model
    else:
        print("training")
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train)





### CODE BEGINS TO EXECUTE

model = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=SAVE_PATH
)

df = pd.read_pickle(FILE_TRAIN)

model.train(
    input_fn=lambda: my_input_fn(df)
)
