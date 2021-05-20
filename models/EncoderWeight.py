from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

tf.enable_eager_execution()
tf.compat.v1.set_random_seed(6603)

print(tf.__version__)
args = parameter_parser()

"""
The graph feature and the pattern feature are fed into the AME network for giving the final detection result and the 
interpretable weights.
"""


class EncoderWeight:
    def __init__(self, graph_train, graph_test, pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
                 pattern3test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        input_dim = tf.keras.Input(shape=(1, 250), name='input')

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.pattern1train = pattern1train
        self.pattern2train = pattern2train
        self.pattern3train = pattern3train
        self.pattern1test = pattern1test
        self.pattern2test = pattern2test
        self.pattern3test = pattern3test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)

        graph2vec = tf.keras.layers.Dense(200, activation='relu', name='outputgraphvec')(input_dim)
        graphweight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputgraphweight')(graph2vec)
        newgraphvec = tf.keras.layers.Multiply(name='outputnewgraphvec')([graph2vec, graphweight])

        pattern1vec = tf.keras.layers.Dense(200, activation='relu', name='outputpattern1vec')(input_dim)
        pattern1weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern1weight')(pattern1vec)
        newpattern1vec = tf.keras.layers.Multiply(name='newpattern1vec')([pattern1vec, pattern1weight])

        pattern2vec = tf.keras.layers.Dense(200, activation='relu', name='outputpattern2vec')(input_dim)
        pattern2weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern2weight')(pattern2vec)
        newpattern2vec = tf.keras.layers.Multiply(name='newpattern2vec')([pattern2vec, pattern2weight])

        pattern3vec = tf.keras.layers.Dense(200, activation='relu', name='outputpattern3vec')(input_dim)
        pattern3weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern3weight')(pattern3vec)
        newpattern3vec = tf.keras.layers.Multiply(name='newpattern3vec')([pattern3vec, pattern3weight])

        mergevec = tf.keras.layers.Concatenate(axis=1, name='mergevec')(
            [newgraphvec, newpattern1vec, newpattern2vec, newpattern3vec])
        flattenvec = tf.keras.layers.Flatten(name='flattenvec')(mergevec)
        finalmergevec = tf.keras.layers.Dense(100, activation='relu', name='outputmergevec')(flattenvec)

        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(finalmergevec)

        model = tf.keras.Model(inputs=[input_dim], outputs=[prediction])

        adama = tf.keras.optimizers.Adam(lr)
        loss = tf.keras.losses.binary_crossentropy
        model.compile(optimizer=adama, loss=loss, metrics=['accuracy'])
        model.summary()

        self.model = model
        self.finalmergevec = finalmergevec

    """
    Training model
    """

    def train(self):
        # create the history instance
        train_history = self.model.fit([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train],
                                       self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                       class_weight=self.class_weight, validation_split=0.1, verbose=2)

        # print('history:')
        # print(str(train_history.history))

        # decoder the training vectors
        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputmergevec').output)
        finalvec_output = finalvec.predict(
            [self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train])
        finalveclayer = tf.keras.layers.Dense(1000, activation='relu')
        finalvec = finalveclayer(finalvec_output)
        finalvecvalue = finalvec.numpy()
        value = np.hsplit(finalvecvalue, 4)
        # print(value)
        print(value[0].shape, value[1].shape, value[2].shape, value[3].shape)

        # self.model.save_weights("model.pkl")

    """
    Testing model
    """

    def test(self):
        # self.model.load_weights("_model.pkl")
        values = self.model.evaluate([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                     self.y_test, batch_size=self.batch_size, verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])

        # graphweight
        graphweight = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputgraphweight').output)

        graphweight_output = graphweight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern1weight
        pattern1weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern1weight').output)
        pattern1weight_output = pattern1weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern2weight
        pattern2weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern2weight').output)
        pattern2weight_output = pattern2weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern3weight
        pattern3weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern3weight').output)
        pattern3weight_output = pattern3weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # output the weights
        print("start")
        gw = graphweight_output.flatten()
        np.savetxt("results/re_gw.txt", gw)
        g_av = gw.mean()
        print("gw_mean:", g_av, "gw_all:", gw.var())
        pw1 = pattern1weight_output.flatten()
        np.savetxt("results/re_pw1.txt", pw1)
        pw1_av = pw1.mean()
        print("pw1_mean:", pw1_av, "pw1_all:", pw1.var())
        pw2 = pattern2weight_output.flatten()
        np.savetxt("results/re_pw2.txt", pw2)
        pw2_av = pw2.mean()
        print("pw2_mean:", pw2_av, "pw2_all:", pw2.var())
        pw3 = pattern3weight_output.flatten()
        np.savetxt("results/re_pw3.txt", pw3)
        pw3_av = pw3.mean()
        print("pw3_mean:", pw3_av, "pw3_all:", pw3.var())
        f = open("results/re_weights.txt", 'a')
        f.write(
            "g_av: " + str(g_av) + ", gw_all :" + str(gw.var()) + "\n pw1_mean:" + str(pw1_av) + ", pw1_all:" + str(
                pw1.var()) + "\n pw2_mean: " + str(pw2_av) + ", pw2_all: " + str(pw2.var()) + "\n pw3_mean: " + str(
                pw3_av) + ", pw3_all: " + str(pw3.var()) + "\n")

        print("end")

        # decoder the testing vectors
        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputmergevec').output)
        finalvec_output = finalvec.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        finalveclayer = tf.keras.layers.Dense(1000, activation='relu')
        finalvec = finalveclayer(finalvec_output)
        finalvecvalue = finalvec.numpy()
        value = np.hsplit(finalvecvalue, 4)
        # print(value)
        # print(value[0].shape, value[1].shape, value[2].shape, value[3].shape)

        # predictions
        predictions = self.model.predict([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                         batch_size=self.batch_size).round()
        print('predict:')
        predictions = predictions.flatten()
        print(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
