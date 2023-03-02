from matplotlib import pyplot as plt #Grafik oluşturmak için
import numpy as np
import pandas as pd
import sympy as sym
from sympy import diff

plot_x2 = []
plot_y2 = []



class ReturnValue():                      #İstediğmiz herhangi bir csv dosyasını rahat bir şekilde okuyabilmek için
    def __init__(self,data,x_data,Species,x):
        self.data = data
        self.Species = Species
        self.x = x
        self.x_data =x_data

def Definition(a):
    data = pd.read_csv(a)
    x_data = np.array(data)# Csv dosyasından alınan datayı array şeklinde numpy ile düzenliyoruz
    Species = np.delete(x_data,0,1)#[0]:SepalLengthCm, [1]:SepalWidthCm, [2]:PetalLengthCm, [3]:PetalWidthCm, [4]: Species

    for j in range(len(Species)):  #Matris içindeki string verileri backward işleminde kullanabilmek için int verilere çeviriyoruz.
        if Species[j,4] == 'Iris-setosa':
            Species[j,4] = 1
        elif Species[j,4] == 'Iris-versicolor':
            Species[j,4] = 2
        elif Species[j,4] == 'Iris-virginica':
            Species[j,4] = 3
    x = np.delete(Species,4,1)#[0]:SepalLengthCm, [1]:SepalWidthCm, [2]:PetalLengthCm, [3]:PetalWidthCm
    return ReturnValue(data,x_data,Species,x) #Returnde class yazarak fonksiyondan birden çok çıktı alabiliyoruz.

class Train():
    def __init__(self,weight,bias,data,yDegeri):
        self.weight = weight
        self.bias = bias
        self.data = data
        self.yDegeri = yDegeri
    
    def ActivationFuncTanh(self,zfunce):
        re = (2/(1+ np.exp((-2)*float(zfunce)))) -1
        re = np.tanh(float(zfunce))
        return re
    
    def ActivationFuncRelu(self,zfunce):
        if zfunce  >= 0:
            re = zfunce
        elif zfunce < 0:
            0 
        return re
    
    def ActivationFuncLeakyRelu(self,zfunce):
        if zfunce.any()  >= 0:
            re = zfunce
        elif zfunce.any() < 0:
            re= 0.01*zfunce
        return re
    
    def ActivationFuncSwish(self,zfunce):
        re = zfunce /(1+np.exp(float(-zfunce)))
        return re

    def ileriYayilim(self):
        z = np.dot(self.weight,self.data) + self.bias
        return z
    
    def backBias(self,L2):
        z = sym.Symbol('z') #Derivative of error
        f = 1/2*(z-self.yDegeri)**2
        df = diff(f,z)
        error = df.subs(z,float(L2)).evalf()
        print(error)
        self.bias = self.bias - 0.01 * error
        return self.bias
    
    def backward(self,z):
        L2 = sym.Symbol('L2') #Derivative of error
        f = 1/2*(L2-self.yDegeri)**2
        df = diff(f,L2)
        error = df.subs(L2,float(z)).evalf()
        print(error)
        #Derivative of activate
        w = sym.Symbol('w')
        f2 = np.dot(self.weight,self.data) + self.bias
        #-------------------------------------
        if z.any() >= 0:
            self.weight = self.weight - 0.01* error *self.data
        elif z < 0:
            self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data*0.01
        return self.weight

    def Error(self,z):
        a = 1/2*(z-self.yDegeri)**2
        return a



class neuralNetwork():
    def __init__(self,csv):
        self.csv = csv
    

    def start(self):
        arg = np.random.default_rng(1) #Random sayı üretmek için
        w = arg.random((3,4))
        w2 = arg.random((1,3))
        bias = arg.random()
        bias2 = arg.random()
        CSV = Definition(self.csv)
        for i in range(len(CSV.data)):              # Tanımladığımız  fonksiyonlara datamızı okutuyoruz.
            R1 = Train(w,bias,CSV.x[i],CSV.Species[i,4])
            R22 = R1.ileriYayilim()
            print('weight',w)
            print('data',CSV.x[i])
            print('bias',bias)
            print(R22)
            R2 = R1.ActivationFuncLeakyRelu(R22)
            print('Activate',R2)
            # R3 = R1.Error(R2)
            # print('Error',R3)
            L1 = Train(w2,bias2,R2,CSV.Species[i,4])
            L22 = L1.ileriYayilim()
            print('L22',L22)
            L2 = L1.ActivationFuncLeakyRelu(L22)
            print('L2',L2)
            L3 = L1.Error(L2)
            print('Error',L3)
            # print('Bias',self.bias)
            print('bias',bias)
            bias = R1.backBias(L2)
            print('last Bias',bias)
            w= R1.backward(R2)
            plot_x2.append(i) #Grafik oluşturabilmek için değerlerimizi liste şeklinde topluyoruz.
            plot_y2.append(L3)
        plt.title("Value of error function")#Grafiğe isim verme
        plt.plot(plot_x2,plot_y2,color ="red")
        #print(w,'111111111111111111111111111')
        weights = pd.DataFrame(w)# Son ulaşılan doğru ağırlıkların csv dosyasına kaydedilmesi
        bias = pd.DataFrame(bias)
        bias.to_csv("/home/enes/Desktop/AESK/nesntesp5/bias.csv")
        weights.to_csv("/home/enes/Desktop/AESK/nesntesp5/weights.csv")
        plt.show()

neuralNetwork('Iris.csv').start()
