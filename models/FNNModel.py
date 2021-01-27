"""
Author: Messi-Q

Date: Created on 11:08 2020-10-22  
"""
from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

args = parameter_parser()

"""
The feed-forward network for extracting the pattern feature
"""


class FNNModel:
    def __init__(self, pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
                 pattern3test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        input_dim = tf.keras.Input(shape=(1, 250), name='input')
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

        pattern1vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern1vec')(input_dim)
        pattern2vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern2vec')(input_dim)
        pattern3vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern3vec')(input_dim)

        mergevec = tf.keras.layers.Concatenate(axis=1, name='mergevec')(
            [pattern1vec, pattern2vec, pattern3vec])  # concatenate patterns
        flattenvec = tf.keras.layers.Flatten(name='flattenvec')(mergevec)  # flatten pattern vectors into one vec

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
        train_history = self.model.fit([self.pattern1train, self.pattern2train, self.pattern3train], self.y_train,
                                       batch_size=self.batch_size, epochs=self.epochs, class_weight=self.class_weight,
                                       validation_split=0.2, verbose=2)

        print('history:')
        print(str(train_history.history))

        # decoder the training vectors
        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputmergevec').output)
        finalvec_output = finalvec.predict([self.pattern1train, self.pattern2train, self.pattern3train])
        finalveclayer = tf.keras.layers.Dense(1000, activation='relu')
        finalvec = finalveclayer(finalvec_output)
        finalvecvalue = finalvec.numpy()
        value = np.hsplit(finalvecvalue, 4)
        print(value)
        print(value[0].shape, value[1].shape, value[2].shape, value[3].shape)

        # self.model.save_weights("model.pkl")

    """
    Testing model
    """

    def test(self):
        # self.model.load_weights("_model.pkl")
        values = self.model.evaluate([self.pattern1test, self.pattern2test, self.pattern3test], self.y_test,
                                     batch_size=self.batch_size, verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])

        # decoder the testing vectors
        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputmergevec').output)
        finalvec_output = finalvec.predict([self.pattern1test, self.pattern2test, self.pattern3test])
        finalveclayer = tf.keras.layers.Dense(1000, activation='relu')
        finalvec = finalveclayer(finalvec_output)
        finalvecvalue = finalvec.numpy()
        value = np.hsplit(finalvecvalue, 4)
        print(value)
        print(value[0].shape, value[1].shape, value[2].shape, value[3].shape)

        # predictions
        predictions = self.model.predict([self.pattern1test, self.pattern2test, self.pattern3test],
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
