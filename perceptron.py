#!/usr/bin/env python3

import numpy as np
import math

epochs = 30
numb_of_iterations = 80
learning_rate = 0.001

def threshold_function(a): #slenkstinė aktyvacijos fukcija
    if a >= 0: return 1
    else: return 0

def sigmoid_function(a): #sigmoidinė aktyvacijos fukcija
    return 1/(1+math.exp(-1*a))

def calculate_error(y, t): #paklaidos apskaičiavimo formulė
    error_of_calculation = 0
    for i in range(y.size):
        error_of_calculation += 0.5 * (y[i] - t[i]) ** 2
    return error_of_calculation


class Perceptron: #neurono inicijavimo klasė
    def __init__(self, activation_function):
        self.function = activation_function
        self.weights = np.random.rand(5)
        self.learnin_rate = learning_rate

    def calculate(self, x):
        output = 0
        for i in range(x.size):
            output += self.weights[i] * x[i] #apskaičiuojame išvestį naudodami formulę suma svoris*slenkstis
        return self.function(output) #naudojame slenkstinę arba sigmodinę aktyvacijos funkciją

    def learn(self, x, y, t):
        error_of_calculation = t - y
        for i in range(self.weights.size):
            if y != t:
               self.weights[i] = self.weights[i] + self.learnin_rate * error_of_calculation * x[i] #keičiame svorio dydį
            else: break


x = np.ones(())
arr = []
arr2 = []
with open ("/Users/einius9/Desktop/iris2main.data", 'r') as myfile: #nuskaitome duomenis iš failo
    for line in myfile:
        x2,x3,x4,x5,label = map(float,line.split(','))
        arr.append([1,x2,x3,x4,x5])
        arr2.append(label)

x = np.array(arr)
y = np.zeros(numb_of_iterations)
t = np.array(arr2)

testx = np.ones(())
arrt = []
arrt2 = []
counter = 0
with open ("/Users/einius9/Desktop/iris2test.data", 'r') as myfile:
    for line in myfile:
        x2,x3,x4,x5,label = map(float,line.split(','))
        arrt.append([1,x2,x3,x4,x5])
        arrt2.append(label)
        counter += 1

testx = np.array(arrt)
testy = np.zeros(counter)
testt = np.array(arrt2)

neuron = Perceptron(sigmoid_function)

for i in range(numb_of_iterations):
    y[i] = neuron.calculate(x[i]) #apskaičiuojame pirminę išvestį

error_of_calculation = calculate_error(y, t) #apskaičiuojame paklaidą

for e in range(epochs): #apmokome neuroną, tai yra, keičiame svorių reikšmes, kol error_of_calculation tampa minimali
    for i in range(numb_of_iterations):
        y[i] = neuron.calculate(x[i]) #apskaičiuojame išvestį
        neuron.learn(x[i], y[i], t[i]) #apmokome neuroną, keičiame svorių dydį

error_of_calculation = calculate_error(y, t)
print(f"weights: {neuron.weights}")

count = 0
for j in range(counter):
    testy[j] = neuron.calculate(testx[j])
    if testy[j] > 0.5: #sigmoidinės funkcijos atveju apvalinam paklaidą
        testy[j] = 1
    else:
        testy[j] = 0

    if testy[j] == testt[j]: #tikrinam, kiek klasių atpažinta teisingai
        count += 1

error_of_calculation = calculate_error(testy, testt) #skaičiuojame paklaidą
print(f"Paklaida po testavimo {error_of_calculation}")
precision = count/counter*100 #skaičiuojame tikslumo matą, kiek procentų atpažinta teisingai
print(f"Klasifikavimo tikslumas {precision}")
