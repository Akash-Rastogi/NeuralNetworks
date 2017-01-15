# Rastogi, Akash
# 1001-408-667
# 2016-10-02
# Assignment_03

import numpy as np
import Tkinter as Tk
import scipy.misc
import os
import matplotlib
import random
from blaze.compute.numpy import epoch

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self):
        # get current working directory
        dir_path = os.getcwd()
        images_path =  dir_path + "/mnist_images"
        
        sample_filenames = []
        sample_imgs_matrix = np.zeros(shape=(1000, 784))
        target_imgs_matrix = np.zeros(shape=(1000, 10))

#         Reading images as vectors
        count = -1
        for file in os.listdir(images_path):
            if file.endswith(".png"):
                count = count + 1 
                sample_image_vector = np.array(self.read_one_image_and_convert_to_vector(images_path + "/" +file))
                sample_filenames = np.append(sample_filenames, file)
                for i in range(len(sample_image_vector)):
                    sample_imgs_matrix[count][i] = sample_image_vector[i]
                
#         Creating Targets according to file names
        count = -1
        for file in os.listdir(images_path):
            count = count + 1;
            target_image_vector = []
            if file.startswith("0_") and file.endswith(".png"):
                target_image_vector = [1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            elif file.startswith("1_") and file.endswith(".png"):
                target_image_vector = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
            elif file.startswith("2_") and file.endswith(".png"):
                target_image_vector = [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1]
            elif file.startswith("3_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
            elif file.startswith("4_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1]
            elif file.startswith("5_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
            elif file.startswith("6_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1]
            elif file.startswith("7_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1]
            elif file.startswith("8_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
            elif file.startswith("9_") and file.endswith(".png"):
                target_image_vector = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
                
            for i in range(len(target_image_vector)):
                target_imgs_matrix[count][i] = target_image_vector[i]
        
#         Shuffling the Samples as well as Targets in same order
        self.sample_shuffler(sample_imgs_matrix, target_imgs_matrix)
        
        
    def sample_shuffler(self, sample_imgs_matrix, target_imgs_matrix):
        #method to shuffle samples and targets
        random_values = random.sample(range(1000), 1000)    # generate unique random numbers in range 1000
        
        shuffled_sample_imgs_matrix = np.zeros(shape=(1000, 784))
        shuffled_target_imgs_matrix = np.zeros(shape=(1000, 10))
        
        count = -1
        for random_value in random_values:
            count = count +1
            shuffled_sample_imgs_matrix[count] = sample_imgs_matrix[random_value]
            shuffled_target_imgs_matrix[count] = target_imgs_matrix[random_value]
        
        self.targets = shuffled_target_imgs_matrix
        self.samples = shuffled_sample_imgs_matrix

    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
        for i in range(len(img)):
            for j in range(len(np.transpose(img))):
                img[i][j] = img[i][j] /255   # Normalization by diving every value by 255
        return img.reshape(-1,1) # reshape to column vector and return it
    
nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights":-1,  # minimum initial weight
    "max_initial_weights": 1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 10
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification,
                    "selected_option" : self.selected_option
                    }
        
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def adjust_weights(self, selected_option):
        for epoch in range(21):
            self.neural_network.adjust_weights(self.data_set.samples, self.data_set.targets, 
                                           self.neural_network.calculate_output(self.data_set.samples), selected_option, epoch)


class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = 0
        self.xmax = 20
        self.ymin = 0
        self.ymax = 100
        self.master.update()
        self.learning_rate = self.nn_experiment.learning_rate
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.output = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("Multiple Linear Classifiers")
        plt.xlabel('Number of Epochs', fontsize=18)
        plt.ylabel('Error Rate', fontsize=18)
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=0.01, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.randomize_weights_button = Tk.Button(self.buttons_frame,
                                                  text="Randomize Weights",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.randomize_weights_button_callback())
        self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        choices = ['Filtered Learning', 'Delta Rule', 'Unsupervised Hebb']
        
        var = Tk.StringVar(self.buttons_frame)
        # initial value
        var.set('Select Rule')

        self.selected_option = Tk.OptionMenu(self.buttons_frame,
                                                    var, *choices, command=lambda *args: self.dropdownselect_callback(var))
        self.selected_option.pack()
        self.selected_option.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
    def dropdownselect_callback(self, var):
        self.selected_option = var.get()
            
    def refresh_display(self):
        self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.refresh_display()

    def adjust_weights_button_callback(self):
        if self.selected_option != 'Filtered Learning' and self.selected_option != "Delta Rule" and self.selected_option != 'Unsupervised Hebb':
            return
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        self.nn_experiment.adjust_weights(self.selected_option)
        self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()
        self.nn_experiment.neural_network.plot_graph()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

neural_network_default_settings = {
    # Optional settings
    "min_initial_weights":-0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10}],  # list of dictionaries
    "selected_option": "Select"
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """
    graph_res_error = np.zeros(shape=(1000, 1))
    
    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights()

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output
    
    def plot_graph(self):
        # plot graph for errors and epoch
        x_vector = np.zeros(shape=(1000, 1))
        y_vector = np.zeros(shape=(1000, 1))
        for i in range(1000):
            x_vector[i] = i
            y_vector[i] = self.graph_res_error[i] * 100
        matplotlib.pyplot.plot(x_vector, y_vector)
        
    def adjust_weights(self, input_samples, target, output, selected_option, epoch):
        #calculate error rate and call to appropriate learning methods
        classified = 0;
        for i in range(1000):
            error = target[i] - output[i]
            
            if selected_option == 'Filtered Learning':
                self.adjust_by_filtered_learning(input_samples[i], target[i])
            elif selected_option == "Delta Rule":
                self.adjust_by_delta_rule(input_samples[i], error)
            elif selected_option == 'Unsupervised Hebb':
                self.adjust_by_unsupervised_hebb(input_samples[i], output[i])
            
            if (target[i] == output[i]).all():
                classified = classified + 1;
                
        error_per_epoch = float(1000-classified) / 1000
        self.graph_res_error[epoch] = error_per_epoch
        print self.graph_res_error[epoch]
            
    def adjust_by_filtered_learning(self, input_samples, target):
#         Initialinz decay factor for Smoothing
        decay = 0.1
        smoothing = 1 - decay
        for layer_index, layer in enumerate(self.layers):
            inputMatrixWithBias = np.append(input_samples, [1])
            product = np.dot(np.transpose(np.matrix(target)), np.matrix(inputMatrixWithBias))
            requiredChangeInWeights = self.learning_rate * product
            for index in range(len(layer.weights)):
                layer.weights[index] = (layer.weights[index]) + requiredChangeInWeights[index]
                
    def adjust_by_delta_rule(self, input_samples, error):
        for layer_index, layer in enumerate(self.layers):
            inputMatrixWithBias = np.append(input_samples, [1])
            product = np.dot(np.transpose(np.matrix(error)), np.matrix(inputMatrixWithBias))
            requiredChangeInWeights = self.learning_rate * product
            for index in range(len(layer.weights)):
                layer.weights[index] = layer.weights[index] + requiredChangeInWeights[index]
        
    def adjust_by_unsupervised_hebb(self, input_samples, output):
        for layer_index, layer in enumerate(self.layers):
            inputMatrixWithBias = np.append(input_samples, [1])
            product = np.dot(np.transpose(np.matrix(output)), np.matrix(inputMatrixWithBias))
            requiredChangeInWeights = self.learning_rate * product
            for index in range(len(layer.weights)):
                layer.weights[index] = layer.weights[index] + requiredChangeInWeights[index]
        
                
single_layer_default_settings = {
    # Optional settings
    "min_initial_weights":-1,  # minimum initial weight
    "max_initial_weights": 1,  # maximum initial weight
    "number_of_inputs_to_layer": 784,  # number of input signals
    "number_of_neurons": 10  # number of neurons in the layer
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """
    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self):
        self.weights = np.random.uniform(self.min_initial_weights, self.max_initial_weights,
                                         (self.number_of_neurons, self.number_of_inputs_to_layer + 1))

        
    def calculate_output(self, input_values):
#         Calculating output vector as -1 or +1
        numOfRows = len(input_values)
        bias = np.ones((numOfRows, 1))
        inputMatrixWithBias = np.c_[input_values, bias]
        net = self.weights.dot(np.transpose(inputMatrixWithBias))
        net = np.transpose(net)
        
        outputs = np.ones(shape=(1000, 10))
        outputs = -1 * outputs
        
#         Setting all values to -1 except the largest, which is set to 1
        for i in range(1000):
            max = np.amax(net[i])
            for j in range(10):
                if net[i][j] == max:
                    outputs[i][j] = 1
        return outputs

if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights":-1,  # minimum initial weight
        "max_initial_weights": 1,  # maximum initial weight
        "number_of_inputs": 784,  # number of inputs to the network
        "learning_rate": 0.001,  # learning rate
        "layers_specification": [{"number_of_neurons": 10}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 10,
        'selected_option': 'Select Rule'
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Linear Associator")
    w, h = main_frame.winfo_screenwidth(), main_frame.winfo_screenheight()
    main_frame.geometry("%dx%d+0+0" % (w, h))
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()