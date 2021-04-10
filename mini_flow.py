import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))




class neuron:
    def __init__(self,size,output_size):
        self.b=[np.random.rand(1,x) for x in size]
        self.w=[[np.random.rand(1,x) for x in size] for y in output_size]
    def forward(self,x):
        self.output=[]
        self.input=x
        for i in self.w:
            self.output.append(sigmoid(np.dot(i,x)+self.b))
        return self.output
    def differentiate(self):
        self.dw=[]
        for i in range(self.output):
            tmp=[]
            for j in range(self.input):
                tmp.append(j*i*(1-i))
            self.dw.append(tmp)
        

class mini_nn:
    def __init__(self, nn_size):
        self.layer=len(nn_size)
        network=[neuron(nn_size[i],nn_size[i+1]) for i in range(len(nn_size)-1)]
    def loss(self,x,y,n):
        error=0
        for neuron in self.network:
            x=neuron.forward(x)
        for i in range(len(y)):
            error+=(1/n)*(x[i]-y[i])**2
        return error
