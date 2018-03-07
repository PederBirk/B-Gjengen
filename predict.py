import numpy
def sigmoid(x):
    return 1/(1+numpy.exp(-x))
W = numpy.random.randn(10,25**2)
bias = numpy.random.randn(10,1)
inp = numpy.random.randn(25**2,1)
a1 = numpy.zeros((10,1))
a1 = sigmoid(numpy.dot(W,inp)-bias)


def NN(layers):
    h= len(layers)-2
    print(str(h) + " hidden layers")
    weights=[]
    biases=[]
    for i in range(1,h+2):
        biases.append(numpy.random.randn(layers[i],1))
        weights.append(numpy.random.randn(layers[i],layers[i-1]))
    return (biases, weights)

        
def predict(inp,nn):
    h = len(nn[0])-1
    a=[sigmoid(numpy.dot(nn[1][0],inp)-nn[0][0])]
    
    for i in range(1,h+1):
        a.append(sigmoid(numpy.dot(nn[1][i],a[i-1])-nn[0][i]))

    return a
