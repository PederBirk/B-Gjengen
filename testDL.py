import dataLoader as dl

dataPath = "C:\\Users\\t_tor\\Unsynced\\extracted_images\\"

symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+','-','y']

num_classes = len(symbols)

#dl.pickleJPEGData(dataPath, symbols)'
data = dl.loadPickledData(dataPath, symbols)