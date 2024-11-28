# skip connection -> helps to preserve spatial info
# output dimentionality smaller than input so we use padding
# Channels increase (helps with capturing new features)

# upsampling substitue of padding

# then increase the resolution
# decrease number of channels
# output is either (0,1) background and foreground


## https://www.geeksforgeeks.org/u-net-architecture-explained/
## Try to implement inception


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

# Custom F1 score metric for TensorFlow/Keras
def f1_metric(y_true, y_pred):
    y_true = tf.round(y_true)  # Round the true values to get 0 or 1
    y_pred = tf.round(y_pred)  # Round the predictions to get 0 or 1
    f1 = tf.py_function(f1_score, (y_true, y_pred), tf.float64)
    return f1

def encoder_block(inputs, num_filters): 

	# Convolution with 3x3 filter followed by ReLU activation 
	x = tf.keras.layers.Conv2D(num_filters, 
							3, 
							padding = 'same')(inputs) 
	x = tf.keras.layers.Activation('relu')(x) 
	
	# Convolution with 3x3 filter followed by ReLU activation 
	x = tf.keras.layers.Conv2D(num_filters, 
							3, 
							padding = 'same')(x) 
	x = tf.keras.layers.Activation('relu')(x) 

	# Max Pooling with 2x2 filter 
	x = tf.keras.layers.MaxPool2D(pool_size = (2, 2), 
								strides = 2)(x) 
	
	return x

def decoder_block(inputs, skip_features, num_filters):
    # Upsample the input (Conv2DTranspose)
    x = tf.keras.layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    
    # Upsample the skip_features to match the upsampled input size (if necessary)
    if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
        skip_features = tf.keras.layers.UpSampling2D(size=(2, 2))(skip_features)
    
    # Merge with skip connection
    x = tf.keras.layers.Concatenate()([x, skip_features])

    # Convolutions after merging
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x



# Unet code 

def unet_model(input_shape = (512, 512, 1), num_classes = 1): 
	inputs = tf.keras.layers.Input(input_shape) 
	
	# Contracting Path 
	s1 = encoder_block(inputs, 512) 
	s2 = encoder_block(s1, 256) 
	s3 = encoder_block(s2, 128) 
	s4 = encoder_block(s3, 32) 
	
	# Bottleneck 
	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(s4) 
	b1 = tf.keras.layers.Activation('relu')(b1) 
	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(b1) 
	b1 = tf.keras.layers.Activation('relu')(b1) 
	
	# Expansive Path 
	s5 = decoder_block(b1, s4, 512) 
	s6 = decoder_block(s5, s3, 256) 
	s7 = decoder_block(s6, s2, 128) 
	s8 = decoder_block(s7, s1, 32) 
	
	# Output 
	outputs = tf.keras.layers.Conv2D(num_classes, 
									1, 
									padding = 'same', 
									activation = 'sigmoid')(s8) 
	
	model = tf.keras.models.Model(inputs = inputs, 
								outputs = outputs, 
								name = 'U-Net') 
	return model 

if __name__ == '__main__': 
    model = unet_model(input_shape=(512, 512, 1), num_classes=1) 
	
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",  # Corrected loss function for binary segmentation
                  metrics=['accuracy', f1_metric]) 

    model.summary()




# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import f1_score

# # Custom F1 score metric for TensorFlow/Keras
# def f1_metric(y_true, y_pred):
#     y_true = tf.round(y_true)  # Round the true values to get 0 or 1
#     y_pred = tf.round(y_pred)  # Round the predictions to get 0 or 1
#     f1 = tf.py_function(f1_score, (y_true, y_pred), tf.float64)
#     return f1

# def encoder_block(inputs, num_filters): 

# 	# Convolution with 3x3 filter followed by ReLU activation 
# 	x = tf.keras.layers.Conv2D(num_filters, 
# 							3, 
# 							padding = 'same')(inputs) 
# 	x = tf.keras.layers.Activation('relu')(x) 
	
# 	# Convolution with 3x3 filter followed by ReLU activation 
# 	x = tf.keras.layers.Conv2D(num_filters, 
# 							3, 
# 							padding = 'same')(x) 
# 	x = tf.keras.layers.Activation('relu')(x) 

# 	# Max Pooling with 2x2 filter 
# 	x = tf.keras.layers.MaxPool2D(pool_size = (2, 2), 
# 								strides = 2)(x) 
	
# 	return x

# def decoder_block(inputs, skip_features, num_filters):
#     # Upsample the input (Conv2DTranspose)
#     x = tf.keras.layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    
#     # Upsample the skip_features to match the upsampled input size (if necessary)
#     if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
#         skip_features = tf.keras.layers.UpSampling2D(size=(2, 2))(skip_features)
    
#     # Merge with skip connection
#     x = tf.keras.layers.Concatenate()([x, skip_features])

#     # Convolutions after merging
#     x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
#     x = tf.keras.layers.Activation('relu')(x)

#     return x



# # Unet code 

# def unet_model(input_shape = (512, 512, 1), num_classes = 1): 
# 	inputs = tf.keras.layers.Input(input_shape) 
	
# 	# Contracting Path 
# 	s1 = encoder_block(inputs, 32) 
# 	s2 = encoder_block(s1, 128) 
# 	s3 = encoder_block(s2, 256) 
# 	s4 = encoder_block(s3, 512) 
	
# 	# Bottleneck 
# 	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(s4) 
# 	b1 = tf.keras.layers.Activation('relu')(b1) 
# 	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(b1) 
# 	b1 = tf.keras.layers.Activation('relu')(b1) 
	
# 	# Expansive Path 
# 	s5 = decoder_block(b1, s4, 512) 
# 	s6 = decoder_block(s5, s3, 256) 
# 	s7 = decoder_block(s6, s2, 128) 
# 	s8 = decoder_block(s7, s1, 32) 
	
# 	# Output 
# 	outputs = tf.keras.layers.Conv2D(num_classes, 
# 									1, 
# 									padding = 'same', 
# 									activation = 'sigmoid')(s8) 
	
# 	model = tf.keras.models.Model(inputs = inputs, 
# 								outputs = outputs, 
# 								name = 'U-Net') 
# 	return model 

# if __name__ == '__main__': 
#     model = unet_model(input_shape=(512, 512, 1), num_classes=1) 
	
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss="binary_crossentropy",  # Corrected loss function for binary segmentation
#                   metrics=['accuracy', f1_metric]) 

#     model.summary()
