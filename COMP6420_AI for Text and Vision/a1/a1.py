import numpy as np
from keras import layers, models


# Task 1:
def image_statistics(image, darkness):
     # Get image resolution (width, height)
     resolution = image.shape[:2] 
     # Count dark pixes in 3 channels based on the darkness threshold
     dark_pixels = tuple(np.sum(image[:, :, i] < darkness)for i in range (3))
     return {'resolution': resolution, 'dark_pixels': dark_pixels}

# Task 2:
def bounding_box(image, top_left, bottom_right):
     # Extract the rows and columns from the tuples 
     top_left_row, top_left_column = top_left
     bottom_right_row, bottom_right_column = bottom_right
     # Return an extract of the image determined by the bounding box 
     return image[top_left_row:bottom_right_row +1, top_left_column:bottom_right_column +1]

# Task 3: 
def build_deep_nn(rows, columns, channels, num_hidden, hidden_sizes, dropout_rates,
                  output_size, output_activation):
     model = models.Sequential() # Initialize the model
     model.add(layers.Flatten(input_shape=(rows, columns, channels)))  
     # Add specified number of hidden layers with relu activation and optional dropout
     for i in range(num_hidden):
        model.add(layers.Dense(hidden_sizes[i], activation='relu'))
        if dropout_rates[i] > 0:
            model.add(layers.Dropout(dropout_rates[i]))
     # Add the final Dense layer with output size and activation function outside the loop
     model.add(layers.Dense(output_size, activation=output_activation))
     return model
        
if __name__ == "__main__":
     import doctest
     doctest.testmod()
