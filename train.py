from matplotlib import pyplot as plt #Grafik oluşturmak için
import numpy as np
import pandas as pd

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

    def activationFunc(self,zfunce): #Farklı aktivasyon fonksiyonlarının tanımlanması 
        #Tanh Function
        # re = (2/(1+ np.exp((-2)*float(zfunce)))) -1
        # re = np.tanh(float(zfunce))
        #Relu func
        # if zfunce  >= 0:
        #     re = zfunce
        # elif zfunce < 0:
        #     0 
        #Leaky Relu Func
        if zfunce.all()  >= 0:
            re = zfunce
        elif zfunce.all() < 0:
            re= np.array([0.01,0.01,0.01])*zfunce
        #Swish Function
        # re = zfunce /(1+np.exp(float(-zfunce)))
        return re
    
    def zfunc(self):    #İleri yayılımda kullanılan işlemler.
        z = np.dot(self.weight,self.data) + self.bias
        return z
    
    def backward(self,z):       #Geri yayılımda yapılan türev işlemleri yerine direkt değerleri girildi.
        #tanh func derivative
        #self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data*(np.tanh(z))**2
        #Relu function
        # if z >= 0:
        #     self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data
        # elif z < 0:
        #     self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data*0
        #Leaky Relu Function
        if z.all() >= 0:
            self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data
        elif z.all() < 0:
            self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data*0.01
        #Swish Function
        # self.weight = self.weight - 0.01*(z -self.yDegeri)*self.data*((1+np.exp(float(-z))*(1+float(z)))/(np.exp(-2*float(z))+1+2*np.exp(float(-z))))
        return self.weight
    
    def backBias(self,z):           #biası da sürekli güncellememiz gerekiyor.
        self.bias = self.bias - 0.01 * (z-self.yDegeri)
        return self.bias
    
    def Error(self,z):              #Hata fonksiyonu ile işlemimizin sonucu ne kadar doğru kontrol ediyoruz.
        a = 1/2*(z-self.yDegeri)**2
        return a

class neuralNetwork():
    def __init__(self,csv):
        self.csv = csv
    

    def start(self):
        arg = np.random.default_rng(1) #Random sayı üretmek için
        w1 = arg.random((1,4))
        # w2 = arg.random((1,4))
        # w3 = arg.random((1,4))
        print('weights',w1)
        bias = arg.random()
        print('bias',bias)
        CSV = Definition(self.csv)
        for i in range(len(CSV.data)):              # Tanımladığımız  fonksiyonlara datamızı okutuyoruz.
            R1 = Train(w1,bias,CSV.x[i],CSV.Species[i,4])
            R22 = R1.zfunc()
            R2 = R1.activationFunc(R22)
            R3 = R1.Error(R2)
            self.bias = R1.backBias(R2)
            print('Activated',R2)
            print('data',CSV.x[i])
            w1 = R1.backward(R2)
            plot_x2.append(i) #Grafik oluşturabilmek için değerlerimizi liste şeklinde topluyoruz.
            plot_y2.append(R3)
        plt.title("Value of error function")#Grafiğe isim verme
        plt.plot(plot_x2,plot_y2,color ="red")
        weights = pd.DataFrame(w1)# Son ulaşılan doğru ağırlıkların csv dosyasına kaydedilmesi
        bias = pd.DataFrame(self.bias)
        bias.to_csv("/home/enes/Desktop/AESK/nesntesp5/bias.csv")
        weights.to_csv("/home/enes/Desktop/AESK/nesntesp5/weights.csv")
        plt.show()

