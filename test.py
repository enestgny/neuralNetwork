from train import Definition
from train import Train
import pandas as pd
import numpy as np

class test():
    def __init__(self,csv,weights):
        self.csv = csv
        self.weights = weights

    def start(self):
        b = 0
        Test = Definition(self.csv)           #Ve sistemimizi tahmin yapabilir duruma getirmiş oluyoruz.
        a = pd.read_csv(self.weights)
        l= pd.read_csv('bias.csv')
        bb= np.array(l)
        bias=np.delete(bb,0,1)
        w = np.array(a)
        ww = np.delete(w,0,1)
        for k in range(len(Test.Species)):
            R1 = Train(ww,bias,Test.x[k],Test.Species[k,4])
            aa =R1.zfunc()
            a = R1.activationFunc(aa)

            if   0 < a < 1.6:
                print(k+1,a,'Iris-setosa',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-setosa': #Hataların olduğunu ve kaç tane olduğunu çıktı olarak göstermesi
                    b += 1
                    print(f'Wrong {b}') 
            elif 1.7 < a < 2.5:
                print(k+1,a,'Iris-versicolor',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-versicolor':
                    b += 1
                    print(f'Wrong {b}') 
            elif 2.5 < a < 4:
                print(k+1,a,'Iris-virginica',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-virginica':
                    b+= 1
                    print(f'Wrong {b}') 

