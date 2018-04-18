from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import dataLoader as dl
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


#Train a CNN using Keras
def trainKerasNetwork():
	batch_size = 128
	epochs = 12
	
	# input image dimensions
	img_rows, img_cols = 45,45
	
	
	#Load training and test data
	dataPath = "C:\\Users\\t_tor\\Unsynced\\extracted_images\\"
	
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+','-','y']
	
	num_classes = len(symbols)
	
	data = dl.loadPickledData(dataPath, symbols)
	
	x_train = data['training-input']
	y_train = data['training-output']
	x_test = data['test-input']
	y_test = data['test-output']
	
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	#Define model
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	
	#Train model
	history = model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
				 shuffle=True,
	          validation_data=(x_test, y_test))
	
	#Evaluate model performance
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return history
	
#Train a CNN using Keras with data augmentation	
def trainKerasNetworkWithDataAugmentation():
	batch_size = 128
	epochs = 30
	
	# input image dimensions
	img_rows, img_cols = 45,45
	
	#Load training and test data
	dataPath = "C:\\Users\\t_tor\\Unsynced\\extracted_images\\"
	
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+','-','y']
	
	num_classes = len(symbols)
	
	data = dl.loadPickledData(dataPath, symbols)
	
	x_train = data['training-input']
	y_train = data['training-output']
	x_test = data['test-input']
	y_test = data['test-output']
	
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	
	#Define model
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	
	#Define data generator using data augmentation
	train_datagen = ImageDataGenerator(
		 rotation_range=20,
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=False)
	
	test_datagen = ImageDataGenerator(rescale=1. / 255)
	
	#Train model
	history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
	                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs, 
	                    validation_data=test_datagen.flow(x_test, y_test, batch_size=batch_size), 
	                    validation_steps=x_test.shape[0]/batch_size)
	
	#Evaluate model performance
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	
	return history

#Save network to file. Yields a json-file for the model definition and a h5-file
#for the model weights.
def dumpNetworkToFile(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("mode.json", "w") as json_file:
	    json_file.write(model_json)
		
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

#Plot the training history
def plotTrainingHistory(history):
  print('Availible variables to plot: {}'.format(history.history.keys()))
  
  #Plot accuracy for training and validation data
  plt.figure(0)
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Training data', 'Validation data'], loc='upper left')
  plt.show()
  
  #Plot loss function for training data and validation data
  plt.figure(1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training data', 'Validation data'], loc='upper left')
  plt.show()