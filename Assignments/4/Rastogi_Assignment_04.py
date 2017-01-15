# Rastogi, Akash
# 1001-408-667
# 2016-10-16
# Assignment_04

import numpy as np
import Tkinter as Tk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Class for UI
class NNGui:
    def __init__(self, master, neuralNetwork):
        self.neural_network = neuralNetwork
        self.master = master
        self.error_to_plot = []  # used for plotting purpose
        
        # x and y axis range
        self.xmin = 0
        self.xmax = 50
        self.ymin = 0
        self.ymax = 1
        self.master.update()
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Least Mean Square Algorithm")
        self.axes = self.figure.add_subplot(111)
        plt.title("Least Mean Square Algorithm")
        plt.xlabel('Batch', fontsize=18)
        plt.ylabel('Error Rate', fontsize=18)
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        
        # Learning Rate Slider
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=0.01, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10)
        self.learning_rate_slider.set(self.neural_network.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Sample Size Slider
        self.sample_size_slider_label = Tk.Label(self.sliders_frame, text="Sample Size Percentage")
        self.sample_size_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10)
        self.sample_size_slider.set(self.neural_network.sample_percent)
        self.sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Batch Size Slider
        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="Batch Size")
        self.batch_size_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=50, to_=500, resolution=50, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10)
        self.batch_size_slider.set(self.neural_network.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W) 
        
        # Delayed Elements Slider
        self.delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="Number of Delayed Elements")
        self.delayed_elements_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10)
        self.delayed_elements_slider.set(self.neural_network.delay)
        self.delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Epoch Slider
        self.epoch_slider_label = Tk.Label(self.sliders_frame, text="Number of Epochs")
        self.epoch_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.epoch_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10)
        self.epoch_slider.set(self.neural_network.epochs)
        self.epoch_slider.bind("<ButtonRelease-1>", lambda event: self.epoch_slider_callback())
        self.epoch_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Set Weights to Zero button
        self.set_weights_zero_button = Tk.Button(self.buttons_frame,
                                                  text="Set Weights to Zero",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.set_weights_zero_button_callback())
        self.set_weights_zero_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Adjust Weights button
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Print Network Settings button
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Print NN Parameters",
                                               bg="yellow", fg="red",
                                               command=lambda: self.print_nnparam_button_callback())
        self.adjust_weights_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
    # CallBack Methods
    
    def learning_rate_slider_callback(self):
        self.neural_network.learning_rate = self.learning_rate_slider.get()
        self.neural_network.set_data_sets()

    def sample_size_slider_callback(self):
        self.neural_network.sample_percent = self.sample_size_slider.get()
        self.neural_network.set_data_sets()

    def batch_size_slider_callback(self):
        self.neural_network.batch_size = self.batch_size_slider.get()
        self.neural_network.set_data_sets()

    def delayed_elements_slider_callback(self):
        self.neural_network.delay = self.delayed_elements_slider.get()
        self.neural_network.set_data_sets()
        self.neural_network.create_new_weights()
        
    def set_weights_zero_button_callback(self):
        self.neural_network.create_new_weights()

    def adjust_weights_button_callback(self):
        self.neural_network.set_data_sets()
        if self.neural_network.number_of_inputs % self.neural_network.batch_size == 0:
            number_of_batch = self.neural_network.number_of_inputs / self.neural_network.batch_size
        else:
            number_of_batch = (self.neural_network.number_of_inputs / self.neural_network.batch_size) + 1
        
        error_code = 0
        for k in range(number_of_batch):
            for j in range(self.neural_network.epochs):
                self.neural_network.epoch_counter = self.neural_network.epoch_counter +1
                for i in range(self.neural_network.batch_size):
                    error_code = self.neural_network.calculate_output(i+1)
                    if error_code != 0:
                        self.neural_network.output_values = self.neural_network.calculate_output(i+1)
                    else:
                        break
                    error_code = self.neural_network.adjust_weights(i+1)
                    if error_code == 0 or error_code == 2:
                        break
                    
            self.error_to_plot.append(self.neural_network.calculate_error_per_batch())
            self.neural_network.batch_counter = self.neural_network.batch_counter + 1
        if error_code == 2:
            print "Plotting Graph for remaining samples"
        else:
            print "Plotting Graph"
        self.plot_graph(number_of_batch)

    def print_nnparam_button_callback(self):
        self.neural_network.print_network_paramenters()
        
    def epoch_slider_callback(self):
        self.neural_network.epochs = self.epoch_slider.get()
        self.neural_network.set_data_sets()

    def plot_graph(self, number_of_batch):
        # plot graph for errors and batch, and Max Errors per batch 
        x_vector = np.zeros(shape=(number_of_batch, 1))
        y_vector = np.zeros(shape=(number_of_batch, 1))
        for i in range(number_of_batch):
            x_vector[i] = i
            y_vector[i] = self.error_to_plot[i] * 100
        matplotlib.pyplot.plot(x_vector, y_vector)
        matplotlib.pyplot.plot(x_vector, self.neural_network.max_error_array, '+r', markersize=10)

# Class for implementing complete network
class NeuralNetwork:
    def __init__(self):
        network_settings = {
            "number_of_inputs": 1367,  # number of inputs to the network
            "learning_rate": 0.001,  # learning rate
            "batch_size": 200, # initial batch size
            "sample_percent": 50, # initial sample size percentage
            "stock_data_size": 2735,  # total size of stock data CSV file provided
            "number_of_neurons": 2,  # number of neurons
            "delay": 5,  #default number of delayed elements
            "epochs": 5  #default number of delayed elements
        }
        self.__dict__.update(network_settings)
        self.set_data_sets()
        self.create_new_weights()
        
    # Method to calculate Error Per Epoch
    def calculate_error_per_batch(self):
        error_array = []
        for i in range(self.batch_size):
            output = self.calculate_output(i+1)
            error_array.append(self.stock_data[i] - output)
        self.max_error_array.append(np.amax(error_array))
        return (np.mean(error_array))*(np.mean(error_array))

    #Method to normalize all inputs        
    def normalize_inputs(self):
        close = []
        volume = []
        for i in range(len(self.stock_data)):
            pair = self.stock_data[i]
            close.append(pair[0])
            volume.append(pair[1])
        
        minClose = np.amin(close)
        maxClose = np.amax(close)
        minVolume = np.amin(volume)
        maxVolume = np.amax(volume)
        
        diff_close = maxClose - minClose
        diff_volume = maxVolume - minVolume
        
        for i in range(len(self.stock_data)):
            pair = self.stock_data[i]
            new_close = (pair[0] - minClose) / diff_close
            new_volume = (pair[1] - minVolume) / diff_volume
            
            pair[0] = new_close
            pair[1] = new_volume
            self.stock_data[i] = pair
        
    # Method to Create New Weights
    # Used when SetWeightToZero button is pressed
    def create_new_weights(self):
        self.network_weights = np.zeros(shape=(self.number_of_neurons, (self.delay*2) + 1))
        
    # Utility Method to print Network Parameters
    def print_network_paramenters(self):
        print "\n\n-------------------------------------------------------------"
        print "number_of_inputs: ", self.number_of_inputs  # number of inputs to the network
        print "learning_rate: ", self.learning_rate  # learning rate
        print "batch_size: ", self.batch_size  # default batch size
        print "Number of Neurons: ", self.number_of_neurons  # Number of Neurons
        print "Number of Epochs: ", self.epochs  # Number of Neurons
        print "Number of Delay Elements: ", self.delay  # Number of Delay Elements
        print "-------------------------------------------------------------\n\n"
               
    # Method to set Data Sets, all value required for network  
    def set_data_sets(self):
        self.delay_counter = 0
        self.epoch_counter = 0
        self.batch_counter = 0
        self.max_error_array = []
        self.stock_data = np.loadtxt('stock_data.csv', skiprows=1, delimiter=',', dtype=np.float32)
        self.normalize_inputs()
        self.stock_data_size = len(self.stock_data)
        self.number_of_inputs = (self.stock_data_size * self.sample_percent) / 100
        self.output_values = np.zeros(2)
        self.error_array = []
        
        if self.number_of_inputs < self.batch_size:
            print "Sample Size can't be less than batch size. Increase Sample Size Percentage."
        else:
            self.targets = self.stock_data[self.sample_percent]

        self.input_values = []
        for i in range(self.delay):  # setting initial input values
            self.input_values.append(self.stock_data[i])
            
    # Method to set new Inputs according to the Delay
    def set_new_inputs(self):
        from_value = self.delay_counter * self.delay
        for i in range(from_value, (self.delay_counter +1) * self.delay):
            self.input_values[i- from_value] = self.stock_data[i]
            
    # Method to Calculate Output
    def calculate_output(self, counter):
        input_array = []
        if counter == self.batch_size:
            return 1
        index = counter + (self.batch_counter*self.batch_size)
        if index == self.number_of_inputs:
            return 0
        for i in range(self.delay):
            value = index + i -1
            if value >= self.stock_data_size:
                print "All Sample cannot be used, as there is delay. Reduce sample size percentage."
                return 2
            pair = self.stock_data[value]
            input_array.append(pair[0])
            input_array.append(pair[1])
        input_array.append(1)
        net = []
        
        for i in range(self.number_of_neurons):
            net.append(self.network_weights[i].dot(np.transpose(input_array)))
        return net
    
    # Method to Adjust Weights
    def adjust_weights(self, counter):
        target_counter = self.delay + counter
        target = self.stock_data[target_counter]
        input_array = []
        
        if counter == self.batch_size:
            return 1
        index = counter + (self.batch_counter*self.batch_size)
        if index == self.number_of_inputs:
            return 0
        for i in range(self.delay):
            value = index + i -1
            if value >= self.stock_data_size:
                print "All Sample cannot be used, as there is delay. Reduce sample size percentage."
                return 2
            pair = self.stock_data[value]
            input_array.append(pair[0])
            input_array.append(pair[1])
        input_array.append(1)

        error = target - self.output_values
        self.error_array.append(error)
        #  Formula: w = w  + 2 a e p
        product1 = 2 * self.learning_rate * error
        product2 = np.outer(np.array(product1), np.array(input_array))
        self.network_weights = self.network_weights + product2
             
if __name__ == "__main__":
    np.random.seed(1)
    neuralNetwork = NeuralNetwork()
    main_frame = Tk.Tk()
    main_frame.title("ADALINE Network")
    w, h = main_frame.winfo_screenwidth(), main_frame.winfo_screenheight()
    main_frame.geometry("%dx%d+0+0" % (w, h))
    ob_nn_gui_2d = NNGui(main_frame, neuralNetwork)
    main_frame.mainloop()