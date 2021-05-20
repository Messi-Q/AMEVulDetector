import numpy as np
from parser import parameter_parser
from models.EncoderWeight import EncoderWeight
from models.EncoderAttention import EncoderAttention
from models.FNNModel import FNNModel
from preprocessing import get_graph_feature, get_pattern_feature

args = parameter_parser()


def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test, label_by_extractor_train, label_by_extractor_valid = get_pattern_feature()

    graph_train = np.array(graph_train)  # The training set of graph feature
    graph_test = np.array(graph_test)  # The testing set of graph feature

    # The training set of patterns' feature
    pattern1train = []
    pattern2train = []
    pattern3train = []
    for i in range(len(pattern_train)):
        pattern1train.append([pattern_train[i][0]])
        pattern2train.append([pattern_train[i][1]])
        pattern3train.append([pattern_train[i][2]])

    # The testing set of patterns' feature
    pattern1test = []
    pattern2test = []
    pattern3test = []
    for i in range(len(pattern_test)):
        pattern1test.append([pattern_test[i][0]])
        pattern2test.append([pattern_test[i][1]])
        pattern3test.append([pattern_test[i][2]])

    # labels of certain contract function in training set (expert annotation)
    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    # The label of certain contract function in testing set (expert annotation)
    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    # labels of pattern feature by a automatic tool
    y_train_pattern = []
    for i in range(len(label_by_extractor_train)):
        y_train_pattern.append(int(label_by_extractor_train[i]))
    y_train_pattern = np.array(y_train_pattern)

    y_test_pattern = []
    for i in range(len(label_by_extractor_valid)):
        y_test_pattern.append(int(label_by_extractor_valid[i]))
    y_test_pattern = np.array(y_test_pattern)

    if args.model == 'EncoderWeight':  # self attention for obtaining the characteristic factor (using the ./pattern_feature/feature_FNN)
        model = EncoderWeight(graph_train, graph_test, np.array(pattern1train), np.array(pattern2train),
                              np.array(pattern3train), np.array(pattern1test), np.array(pattern2test),
                              np.array(pattern3test), y_train, y_test)
    elif args.model == 'EncoderAttention':  # cross attention for computing the attention weight (using the ./pattern_feature/feature_FNN)
        model = EncoderAttention(graph_train, graph_test, np.array(pattern1train), np.array(pattern2train),
                                 np.array(pattern3train), np.array(pattern1test), np.array(pattern2test),
                                 np.array(pattern3test), y_train, y_test)
    elif args.model == 'FNNModel':  # extract pattern feature using a feed-forward network (using the ./pattern_feature/feature_zeropadding)
        model = FNNModel(np.array(pattern1train), np.array(pattern2train), np.array(pattern3train),
                         np.array(pattern1test), np.array(pattern2test), np.array(pattern3test), y_train_pattern,
                         y_test_pattern)

    model.train()  # training
    model.test()  # testing


if __name__ == "__main__":
    main()
