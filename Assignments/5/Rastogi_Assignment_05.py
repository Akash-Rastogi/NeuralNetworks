# Rastogi, Akash
# 1001-408-667
# 2016-11-15
# Assignment_05

import os
import sys
import gc
import theano
import numpy as np
import matplotlib
import scipy.misc
import theano.tensor as T
import random
from time import gmtime, strftime

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Class for creating DataSets
class DataSet:
    def __init__(self, network_settings):
        self.__dict__.update(network_settings)
        self.train_data_filenames = []
        self.test_data_filenames = []

        self.set_data_sets()

    # Method to set Data Sets(all values required for network)
    def set_data_sets(self):
        if self.type == "cifar_100":
            self.train_sample_imgs_matrix = np.zeros(shape=(1000, 3072))
            self.train_target_imgs_matrix = np.zeros(shape=(1000, 10))

            self.test_sample_imgs_matrix = np.zeros(shape=(1000, 3072))
            self.test_target_imgs_matrix = np.zeros(shape=(1000, 10))

        elif self.type == "cifar_1000":
            self.train_sample_imgs_matrix = np.zeros(shape=(10000, 3072))
            self.train_target_imgs_matrix = np.zeros(shape=(10000, 10))

            self.test_sample_imgs_matrix = np.zeros(shape=(1000, 3072))
            self.test_target_imgs_matrix = np.zeros(shape=(1000, 10))

        self.read_images()

    # Method to read all images according to ath provided(here, Current Working Directory)
    def read_images(self):
        # get current working directory
        dir_path = os.getcwd()

        if self.type == "cifar_100":
            train_images_path = dir_path + "/cifar_data_100_10/train"
            test_images_path = dir_path + "/cifar_data_100_10/test"
        elif self.type =="cifar_1000":
            train_images_path = dir_path + "/cifar_data_1000_100/train"
            test_images_path = dir_path + "/cifar_data_1000_100/test"

        count = -1
        for file in os.listdir(train_images_path):
            if file.endswith(".png"):
                count = count + 1
                img_vector = np.array(self.read_one_image_and_convert_to_vector(train_images_path + "/" + file))
                self.train_data_filenames = np.append(self.train_data_filenames, file)
                for i in range(len(img_vector)):
                    self.train_sample_imgs_matrix[count][i] = img_vector[i]
        count = -1
        for file in os.listdir(test_images_path):
            if file.endswith(".png"):
                count = count + 1
                img_vector = np.array(self.read_one_image_and_convert_to_vector(test_images_path + "/" + file))
                self.test_data_filenames = np.append(self.test_data_filenames, file)
                for i in range(len(img_vector)):
                    self.test_sample_imgs_matrix[count][i] = img_vector[i]

        count = -1
        for file in os.listdir(train_images_path):
            count = count + 1;
            target_image_vector = []
            if file.startswith("0_") and file.endswith(".png"):
                target_image_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("1_") and file.endswith(".png"):
                target_image_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("2_") and file.endswith(".png"):
                target_image_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("3_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif file.startswith("4_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif file.startswith("5_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif file.startswith("6_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif file.startswith("7_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif file.startswith("8_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif file.startswith("9_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            for i in range(len(target_image_vector)):
                self.train_target_imgs_matrix[count][i] = target_image_vector[i]

        count = -1
        for file in os.listdir(test_images_path):
            count = count + 1;
            target_image_vector = []
            if file.startswith("0_") and file.endswith(".png"):
                target_image_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("1_") and file.endswith(".png"):
                target_image_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("2_") and file.endswith(".png"):
                target_image_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif file.startswith("3_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif file.startswith("4_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif file.startswith("5_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif file.startswith("6_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif file.startswith("7_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif file.startswith("8_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif file.startswith("9_") and file.endswith(".png"):
                target_image_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            for i in range(len(target_image_vector)):
                self.test_target_imgs_matrix[count][i] = target_image_vector[i]

        # Shuffling the Samples as well as Targets in same order
        self.sample_shuffler()

    # Method to shuffle samples and targets
    def sample_shuffler(self):
        if self.type == "cifar_100":
            train_random_values = random.sample(range(1000), 1000)  # generate unique random numbers in range 1000
            test_random_values = random.sample(range(100), 100)  # generate unique random numbers in range 100

            shuffled_train_sample_imgs_matrix = np.zeros(shape=(1000, 3072))
            shuffled_train_target_imgs_matrix = np.zeros(shape=(1000, 10))

            shuffled_test_sample_imgs_matrix = np.zeros(shape=(100, 3072))
            shuffled_test_target_imgs_matrix = np.zeros(shape=(100, 10))
        elif self.type == "cifar_1000":
            train_random_values = random.sample(range(10000), 10000)  # generate unique random numbers in range 10000
            test_random_values = random.sample(range(1000), 1000)  # generate unique random numbers in range 1000

            shuffled_train_sample_imgs_matrix = np.zeros(shape=(10000, 3072))
            shuffled_train_target_imgs_matrix = np.zeros(shape=(10000, 10))

            shuffled_test_sample_imgs_matrix = np.zeros(shape=(1000, 3072))
            shuffled_test_target_imgs_matrix = np.zeros(shape=(1000, 10))

        count = -1
        for random_value in train_random_values:
            count = count + 1
            shuffled_train_sample_imgs_matrix[count] = self.train_sample_imgs_matrix[random_value]
            shuffled_train_target_imgs_matrix[count] = self.train_target_imgs_matrix[random_value]

        count = -1
        for random_value in test_random_values:
            count = count + 1
            shuffled_test_sample_imgs_matrix[count] = self.test_sample_imgs_matrix[random_value]
            shuffled_test_target_imgs_matrix[count] = self.test_target_imgs_matrix[random_value]

        self.train_targets = shuffled_train_target_imgs_matrix
        self.train_samples = shuffled_train_sample_imgs_matrix

        self.test_targets = shuffled_test_target_imgs_matrix
        self.test_samples = shuffled_test_sample_imgs_matrix

    # Method to read image and return converted vector form(after normalization)
    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        img_vector =[]

        for i in range(len(img)):
            for j in range(len(img)):
                for k in range(3):
                    img_vector.append((img[i][j][k])/255)  # Normalization by diving every value by 255
        return img_vector


# Class for implementing complete network
class NeuralNetwork:
    def __init__(self, network_settings):
        self.__dict__.update(network_settings)

        self.layer_one = self.create_layer_one(network_settings)
        self.layer_two = self.create_layer_two(network_settings)

        if self.type == "cifar_100":
            self.num_of_training_samples = 1000
            self.num_of_testing_samples = 100
        elif self.type == "cifar_1000":
            self.num_of_training_samples = 10000
            self.num_of_testing_samples = 1000

        self.data_set = DataSet(network_settings)

    # Method used for all training and predicting using theano
    def train_by_theano(self, case_number):
        self.training_errors_per_epoch = []
        self.loss_array = []
        confusion_matrix = np.zeros(shape=(10, 10))

        # initialize the bias term
        bias_one = theano.shared(0.0, name="b1")
        bias_two = theano.shared(0.0, name="b2")

        # creating shared variables
        layer_one_weights = theano.shared(self.layer_one.network_weights, name="w1")
        layer_two_weights = theano.shared(self.layer_two.network_weights, name="w2")

        # l2 regulaizer
        l2_regularizer = (
            (layer_one_weights ** 2).sum()
            + (layer_two_weights ** 2).sum()
        )

        # not used currently
        l1_regularizer = (
            abs(layer_two_weights).sum()
            + abs(layer_two_weights).sum()
        )

        # Define some placeholders for our inputs/targets
        input_ph = T.dmatrix('input')
        target_ph = T.dmatrix('target')

        # calculating NetValues
        net_value_1 = T.dot(layer_one_weights, input_ph) + bias_one
        if self.activation_function_case == 1:
            output_1 = T.nnet.relu(net_value_1)
        else:
            output_1 = T.nnet.sigmoid(net_value_1)

        net_value_2 = (T.dot(layer_two_weights, output_1) + bias_two).T
        if self.activation_function_case == 1:
            output_2 = T.nnet.softmax(net_value_2)
        else:
            output_2 = T.nnet.sigmoid(net_value_2)

        prediction = output_2[0].argmax()

        # calculating loss
        loss = T.mean(T.nnet.categorical_crossentropy(output_2, target_ph.T) + ((l2_regularizer * self.l2_lambda)/2))  #
        grad_w1, grad_b1, grad_w2, grad_b2 = T.grad(cost=loss, wrt=[layer_one_weights, bias_one, layer_two_weights, bias_two])

        updates_w2 = [layer_two_weights, layer_two_weights - (grad_w2 * self.learning_rate)]
        updates_b2 = [bias_two, bias_two - (grad_b2 * self.learning_rate)]

        updates_w1 = [layer_one_weights,  layer_one_weights - (grad_w1 * self.learning_rate)]
        updates_b1 = [bias_one, bias_one - (grad_b1 * self.learning_rate)]

        # train method
        train = theano.function([input_ph, target_ph], [loss, prediction],
                                updates=[updates_w1, updates_b1, updates_w2, updates_b2])

        # predict method, for test data set only
        predict = theano.function([input_ph], prediction)

        for epoch in range(self.epochs):  # Epoch
            correctly_classified = 0
            loss_per_epoch = []
            for i in range(self.num_of_training_samples):
                target = self.data_set.train_targets[i:i + 1, :].T
                sample = self.data_set.train_samples[i:i + 1, :].T

                loss, prediction_index = train(sample, target)

                target_vector = self.data_set.train_targets[i]
                target_index = target_vector.argmax()

                if prediction_index == target_index:
                    correctly_classified += 1
                loss_per_epoch.append(loss)

            error = (self.num_of_training_samples - correctly_classified)

            # saving loss & error to plot
            self.loss_array.append(np.mean(loss_per_epoch))
            self.training_errors_per_epoch.append(error)

            print "Epoch #", str(epoch+1), ", Error(misclassified) :", str(error), " , Loss : ", str(np.mean(loss_per_epoch))
            print "======================================"

        correctly_classified = 0
        for i in range(self.num_of_testing_samples):
            sample = self.data_set.test_samples[i:i + 1, :].T

            prediction_index = predict(sample)

            target_vector = self.data_set.test_targets[i]
            target_index = target_vector.argmax()

            confusion_matrix[prediction_index, target_index] += 1

            if prediction_index == target_index:
                correctly_classified += 1

        error = (self.num_of_testing_samples - correctly_classified)
        print "Testing Epoch Error(misclassified) :", str(error), "out of " + str(self.num_of_testing_samples)
        print "======================================"

        print("===================Confusion Matrix for Test Case #: " + str(case_number) + "=====================")
        print("::Prediction X Actual::")
        print(confusion_matrix)
        print("===============================================================================")

        return self.training_errors_per_epoch, self.loss_array

    # Method to create layer one
    def create_layer_one(self, network_settings):

        layer_one_setting = {
            "number_of_neurons": self.num_of_nodes_in_hidden,  # number of neurons in layer
            "number_of_inputs_to_layer": 3072,  # number of inputs to layer
        }
        layer_one = NNLayer(network_settings, layer_one_setting)
        layer_one.create_new_weights()
        return layer_one

    # Method to create layer one
    def create_layer_two(self, network_settings):
        layer_two_setting = {
            "number_of_neurons": 10,  # number of neurons in layer
            "number_of_inputs_to_layer": self.num_of_nodes_in_hidden,  # number of inputs to layer
        }
        layer_two = NNLayer(network_settings, layer_two_setting)
        layer_two.create_new_weights()
        return layer_two

    # Method to plot graph for errors and epoch
    def plot_error_graph(self, case_number, errors_list):

        fig = plt.figure()
        plt.title("Backpropogation using Theano")
        plt.ylim(0,100)
        plt.xlabel('Number of Epochs', fontsize=18)
        plt.ylabel('Error Rate', fontsize=18)

        labels_1 = ['100', '200', '300', '400', '500']
        labels_2 = ['0.1', '0.2', '0.3', '0.4', '0.5']


        if case_number == 6:
            for list, label in zip(errors_list, labels_1):
                if len(list) > 0:
                    plt.plot([x/10 for x in list], label=label)
        elif case_number == 11:
            for list, label in zip(errors_list, labels_2):
                if len(list) > 0:
                    plt.plot([x/10 for x in list], label=label)
        else:
            for list, label in zip(errors_list, labels_2):
                if len(list) > 0:
                    plt.plot([x/10 for x in list])

        plt.legend()

        # creating separate directory for saving all plots
        directory = os.getcwd() + "/Error_Graphs/"
        if not os.path.exists(directory):
            os.makedirs(directory)


        file_name = "Error_Graphs/Error_Case#_" + str(case_number) + ".png"
        fig.savefig(file_name, dpi=fig.dpi)

    # Method to plot graph for loss and epoch
    def plot_loss_graph(self, case_number, loss_list):

        fig = plt.figure()
        plt.title("Backpropogation using Theano")
        plt.xlabel('Number of Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)

        labels_1 = ['100', '200', '300', '400', '500']
        labels_2 = ['0.1', '0.2', '0.3', '0.4', '0.5']

        if case_number == 6:
            for list, label in zip(loss_list, labels_1):
                if len(list) > 0:
                    plt.plot(list, label=label)
        elif case_number == 11:
            for list, label in zip(loss_list, labels_2):
                if len(list) > 0:
                    plt.plot(list, label=label)
        else:
            for list, label in zip(loss_list, labels_2):
                if len(list) > 0:
                    plt.plot(list)

        plt.legend()

        # creating separate directory for saving all plots
        directory = os.getcwd() + "/Loss_Graphs/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = "Loss_Graphs/Loss_Case#_" + str(case_number) + ".png"
        fig.savefig(file_name, dpi=fig.dpi)

# Class to create Network Layers and their weight matrices
class NNLayer:
    def __init__(self, network_settings, layer_one_setting):
        self.__dict__.update(network_settings)
        self.__dict__.update(layer_one_setting)

    def create_new_weights(self):
        self.network_weights = np.random.uniform(self.min_initial_weights, self.max_initial_weights,
                        (self.number_of_neurons, self.number_of_inputs_to_layer))

# Main method
if __name__ == "__main__":
    np.random.seed(1)
    training_network_settings = {
        "number_of_layers": 3,  # inclusive of input and output
        "number_of_outputs": 10,  # number of outputs to the network
        "learning_rate": 0.0001,  # learning rate
        "image_vector_size": 3072,  # single image vector size
        "epochs": 100,  # iterations to train
        "type": "cifar_100",  # or "cifar_1000" : type of data set
        "min_initial_weights": -0.1,  # minimum initials weights for both layers
        "max_initial_weights": 0.1,  # maximum initials weights for both layers
        "l2_lambda": 0.0,  # lambda for regularization
        "activation_function_case": 1, # 1 for (relu, softmax), else (sigmoid, sigmoid) respectively
        "num_of_nodes_in_hidden": 100
    }

    # number of scenarios provided
    number_of_cases = 13

    # changing standard output to file in current working directory
    print("Changing Standard Output Device to log.txt file in Current Working Directory......")
    sys.stdout = open('log.txt', 'w')

    # logging current time
    print("Logging Starts at : " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    errors_list_to_plot = [[] for x in xrange(number_of_cases)]
    loss_list_to_plot = [[] for x in xrange(number_of_cases)]
    index_counter = 0
    for i in range(number_of_cases):
        # Task 1 (i==0)

        # Task 2
        if i == 1:
            training_network_settings.__setitem__("activation_function_case", 2)

        # Task 3
        elif i == 2:
            training_network_settings.__setitem__("activation_function_case", 1)
            training_network_settings.__setitem__("num_of_nodes_in_hidden", 100)
        elif i == 3:
            training_network_settings.__setitem__("num_of_nodes_in_hidden", 200)
        elif i == 4:
            training_network_settings.__setitem__("num_of_nodes_in_hidden", 300)
        elif i == 5:
            training_network_settings.__setitem__("num_of_nodes_in_hidden", 400)
        elif i == 6:
            training_network_settings.__setitem__("num_of_nodes_in_hidden", 500)

        # Task 4
        elif i == 7:
            training_network_settings.__setitem__("epochs", 50) # Reducing number of Epochs as hidden layer nodes are 500 now
            training_network_settings.__setitem__("l2_lambda", 0.1)
        elif i == 8:
            training_network_settings.__setitem__("l2_lambda", 0.2)
        elif i == 9:
            training_network_settings.__setitem__("l2_lambda", 0.3)
        elif i == 10:
            training_network_settings.__setitem__("l2_lambda", 0.4)
        elif i == 11:
            training_network_settings.__setitem__("l2_lambda", 0.5)

        # Task 5
        elif i == 12:
            training_network_settings.__setitem__("l2_lambda", 0.2) # best suitable
            training_network_settings.__setitem__("epochs", 25)  # Reducing number of Epochs as hidden layer nodes are 500 now
            training_network_settings.__setitem__("type", "cifar_1000")

        print("********************************* CASE #" + str(i) + " *********************************")
        print("===========Training Parameters===========")
        print(training_network_settings)
        print("===========Training Parameters===========")
        neural_network = NeuralNetwork(training_network_settings)

        training_errors_per_epoch, loss_array = neural_network.train_by_theano(i)

        errors_list_to_plot.insert(index_counter, training_errors_per_epoch)
        loss_list_to_plot.insert(index_counter, loss_array)

        index_counter += 1
        if i==0 or i==1 or i==6 or i==11 or i==12:
            neural_network.plot_error_graph(i, errors_list_to_plot)
            neural_network.plot_loss_graph(i, loss_list_to_plot)
            errors_list_to_plot = [[] for x in xrange(number_of_cases)]
            loss_list_to_plot = [[] for x in xrange(number_of_cases)]
            index_counter = 0
        elif i>12:
            break
        gc.collect() # calling garbage collector explicitly
    print("*******************************************************************************")

    # logging current time
    print("Logging Stops at : " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))