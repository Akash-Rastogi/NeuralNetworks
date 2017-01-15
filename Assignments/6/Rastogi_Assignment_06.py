# Rastogi, Akash
# 1001-408-667
# 2016-12-02
# Assignment_06

import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import linalg as LA

# References :
# Keras : https://blog.keras.io/building-autoencoders-in-keras.html
# ImageGrid : http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
# PCA : http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python

class DataSet:
    # Method to read all images according to ath provided(here, Current Working Directory)
    def __init__(self):
        # get current working directory
        dir_path = os.getcwd()
        self.train_data_filenames = []
        self.train_sample_imgs_matrix = np.zeros(shape=(20000, 784))

        self.test_data_filenames = []
        self.test_sample_imgs_matrix = np.zeros(shape=(2000, 784))

        self.test_data_filenames2 = []
        self.test_sample_imgs_matrix2 = np.zeros(shape=(100, 784))

        train_images_path = dir_path + "/data_20k"
        test_images_path = dir_path + "/data_2k"
        test_images_path2 = dir_path + "/data_100"

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
        for file in os.listdir(test_images_path2):
            if file.endswith(".png"):
                count = count + 1
                img_vector = np.array(self.read_one_image_and_convert_to_vector(test_images_path2 + "/" + file))
                self.test_data_filenames2 = np.append(self.test_data_filenames2, file)
                for i in range(len(img_vector)):
                    self.test_sample_imgs_matrix2[count][i] = img_vector[i]


    # Method to read image and return converted vector form(after normalization)
    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        img_vector = []

        for i in range(len(img)):
            for j in range(len(img)):
                img_vector.append((img[i][j]) / 255)  # Normalization by diving every value by 255
        return img_vector

# Class for implementing complete network
class NeuralNetwork:
    def __init__(self, network_settings, data_set):
        self.__dict__.update(network_settings)
        self.data_set = data_set

    # Method used for all training and predicting using Keras
    def train_by_keras(self, case_number):
        training_loss = []
        validation_loss = []

        # this is the size of our encoded representations
        encoding_dim = self.num_of_nodes_in_hidden  # 100 floats -> compression of factor 78.4, assuming the input is 784 floats

        # this is our input placeholder
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(784, activation='linear')(encoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input=input_img, output=decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input=input_img, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

        # (x_train, _), (x_test, _) = mnist.load_data()
        x_train = self.data_set.train_sample_imgs_matrix
        x_test = self.data_set.test_sample_imgs_matrix

        # for Callbacks
        history = History()

        if case_number == 0 or case_number == 6:
            for i in range(self.epochs):
                print('================================================== Case #', case_number, 'Epoch #', (i+1), '==================================================')
                autoencoder.fit(x_train, x_train, shuffle=True, validation_data=(x_test, x_test), nb_epoch=1, callbacks=[history])
                training_loss.append(history.history['loss'])
                validation_loss.append(history.history['val_loss'])

        else:
            for i in range(self.epochs):
                print('================================================== Case #', case_number, 'Epoch #', (i+1), '==================================================')
                autoencoder.fit(x_train, x_train, shuffle=True, nb_epoch=1, callbacks=[history])
                training_loss.append(history.history['loss'])
            x_test = self.data_set.test_sample_imgs_matrix
            # score = autoencoder.predict(x_test)
            # validation_loss = score

            score = autoencoder.evaluate(x_test, x_test, verbose=1)
            validation_loss = score
            # print('Test loss:', score[0])


        # Task 4 and 5 goes here
        if case_number == 6:
            x_test = self.data_set.test_sample_imgs_matrix2
            encoded_imgs = encoder.predict(x_test)
            decoded_imgs = decoder.predict(encoded_imgs)
            self.plot_img_matrix(x_test, 7)
            self.plot_img_matrix(decoded_imgs, 8)

            reduced_samples, eigenvalues_samples, eigenvectors_samples = self.get_PCA(self.data_set.test_sample_imgs_matrix, 100)
            self.plot_img_matrix(eigenvectors_samples.T, 9)

            encoded_imgs = encoder.predict(self.data_set.test_sample_imgs_matrix)
            decoded_imgs = decoder.predict(encoded_imgs)

            reduced_decoded, eigenvalues_decoded, eigenvectors_decoded = self.get_PCA(decoded_imgs, 100)
            self.plot_img_matrix(eigenvectors_decoded.T, 10)

        return training_loss, validation_loss, decoder_layer.get_weights()

    # Method used to get PCA, EigenValues and EigenVectors
    def get_PCA(self, data, dims_rescaled_data):

        m, n = data.shape

        # mean center the data
        data -= np.mean(data, axis=0)

        # calculate the covariance matrix
        cov_matrix = np.cov(data, rowvar=False)

        # calculate eigenvectors & eigenvalues of the covariance matrix
        evals, evecs = LA.eigh(cov_matrix)

        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # sort eigenvectors according to same index
        evals = evals[idx]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]

        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs

    # Method used to plot Graph
    def plot_graph(self, case_number, training_loss, validation_loss):
        plt.clf()
        fig = plt.figure()
        label_train = 'Training Loss'
        label_validation = 'Validation Loss'

        plt.title("AutoEncoder using Keras")
        plt.xlabel('Number of Epochs', fontsize=18)
        plt.ylabel('Loss : Mean Squared Error', fontsize=18)


        if case_number==5:
            plt.xlim(0, 110)
            plt.ylim(0, 0.1)
            plt.xlabel('Number of nodes in Hidden Layer', fontsize=18)

            plt.plot([20,40,60,80,100],training_loss, label=label_train, linestyle='None', marker='o', color='b')
            plt.plot([20,40,60,80,100],validation_loss, label=label_validation, linestyle='None', marker='v', color='r')
        else:
            plt.ylim(0, 0.2)
            plt.plot(training_loss, label=label_train)
            plt.plot(validation_loss, label=label_validation)



        plt.legend()
        # plt.show()

        # creating separate directory for saving all plots
        directory = os.getcwd() + "/Outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = "Outputs/Case#_" + str(case_number) + ".png"
        fig.savefig(file_name, dpi=fig.dpi)
        plt.close(fig)

    # Method used to plot Image Grid
    def plot_img_matrix(self, img_matrix, case_number):
        print(img_matrix.shape)
        plt.clf()
        fig = plt.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                         axes_pad=0.05,  # pad between axes in inch.
                         )

        for i in range(100):
            grid[i].imshow(img_matrix[i].reshape(28, 28), cmap='Greys_r')  # The AxesGrid object work as a list of axes.
            grid[i].get_xaxis().set_visible(False)
            grid[i].get_yaxis().set_visible(False)

        # creating separate directory for saving all plots
        directory = os.getcwd() + "/Outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = "Outputs/Case#_" + str(case_number) + ".png"
        fig.savefig(file_name, dpi=fig.dpi)
        plt.close(fig)

# Main method
if __name__ == "__main__":
    np.random.seed(1)

    # Initial Network Setting
    network_settings = {
        "epochs": 50,  # number of epochs
        "num_of_nodes_in_hidden": 100
    }

    # Creating dataset for Network
    data_set = DataSet()

    # number of scenarios provided
    number_of_cases = 6

    # For task 2
    training_loss_task2 = []
    validation_loss_task2 = []

    # Call to all tasks all together
    for case_number in range(number_of_cases+1):
        if case_number == 0:
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            neural_network.plot_graph(case_number, training_loss, validation_loss)

        if case_number == 1:
            network_settings.__setitem__("num_of_nodes_in_hidden", 20)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            training_loss_task2.append(np.mean(training_loss))
            validation_loss_task2.append(validation_loss)

        elif case_number == 2:
            network_settings.__setitem__("num_of_nodes_in_hidden", 40)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            training_loss_task2.append(np.mean(training_loss))
            validation_loss_task2.append(validation_loss)

        elif case_number == 3:
            network_settings.__setitem__("num_of_nodes_in_hidden", 60)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            training_loss_task2.append(np.mean(training_loss))
            validation_loss_task2.append(validation_loss)

        elif case_number == 4:
            network_settings.__setitem__("num_of_nodes_in_hidden", 80)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            training_loss_task2.append(np.mean(training_loss))
            validation_loss_task2.append(validation_loss)

        elif case_number == 5:
            network_settings.__setitem__("num_of_nodes_in_hidden", 100)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            training_loss_task2.append(np.mean(training_loss))
            validation_loss_task2.append(validation_loss)
            neural_network.plot_graph(case_number, training_loss_task2, validation_loss_task2)

        elif case_number == 6:
            network_settings.__setitem__("epochs", 100)
            neural_network = NeuralNetwork(network_settings, data_set)
            training_loss, validation_loss, weights = neural_network.train_by_keras(case_number)
            neural_network.plot_img_matrix(weights[0], case_number) # Task 3