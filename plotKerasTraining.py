%matplotlib inline
from matplotlib import pyplot as plt

def plot_training_score(history):
  #print('Availible variables to plot: {}'.format(history.history.keys()))
  
  plt.figure()
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.legend(['Training', 'Validation'])

  plt.figure()
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.legend(['Training', 'Validation'], loc='lower right')
  plt.show()