import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))




class neuron:
    def __init__(self,size,output_size):
        self.b=[np.random.rand(1,x) for x in output_size]
        self.w=[[np.random.rand(1,x) for x in size] for y in output_size]
    def forward(self,x):
        self.output=[]
        self.input=x
        for i in range(len(self.w)):
            self.output.append(sigmoid(np.dot(self.w[i],x)+self.b[i]))
        return self.output
    def differentiate(self):
        self.dw=[]
        self.db=[]
        self.dx=[0 for i in self.input]
        for i in self.output:
            self.db.append(i*(1-i))
        for i in self.db:
            tmp=[]
            for j in self.input:
                tmp.append(j*i)
            self.dw.append(tmp)
        for i in self.w:
            for j in range(len(i)):
                self.dx[j]+=i[j]*self.db[j]
    def prev_diffs(self,dx):
        tmp=0
        for i in range(len(dx)):
            self.db[i]*=dx[i]
            self.dw[i]=[k*dx[i] for k in self.dw[i]]
            tmp+=dx[i]
        self.dx=[j*tmp for j in self.dx]


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
