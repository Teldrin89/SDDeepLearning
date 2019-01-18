# In order to increase the accuracy of the model (and reduce loss) it's important to tweak specific parameters
# and find the best solution. One of the area to explore is the description and number of layers - here is the
# base setup for that kind of parametrization

# import time for name of models
import time

# run the loop for few dense layers number
dense_layers = [0, 1, 2]
# check different number of layer sizes
layer_sizes = [32, 64, 128]
# since the 64 size was working it is a good practice to check the 2 neighbouring values (a half value and twice one)
# the pick one based on the accuracy and loss changes and comparison between the models
# check different number of convolutional layers - has to be at least 1
conv_layers = [1, 2, 3]

# run 3 for loops
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # prepare a name structure for models (using all parameters plus time)
            NAME = "{}-dense-{}-nodes-{}-conv-{}".format(dense_layer, layer_size, conv_layer, int(time.time()))
            # printout all names
            print(NAME)
