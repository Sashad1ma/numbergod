#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      mozzz
#
# Created:     24.08.2018
# Copyright:   (c) mozzz 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

class Data():
    def __init__ (self,data_file_name,labels_file_name,byte_number=None):
        #Считываем байты из файла с картинками
        in_file = open(data_file_name,"rb")
        self.pictures = in_file.read(byte_number)[16:]
        in_file.close()
        #Считываем байты из файла с подписями
        in_file = open(labels_file_name,"rb")
        self.lables = in_file.read()[8:]
        in_file.close()
        #Создаём словарь с дата сетами
        pixels_lst = []
        label_plus_picture = []
        self.data_set = {}
        self.set_number = 0
        for i in self.pictures:
            pixels_lst.append(i)
            if len(pixels_lst) == 784:
                label_plus_picture.append(self.lables[self.set_number])
                label_plus_picture.append(pixels_lst)
                self.data_set[self.set_number] = copy.deepcopy(label_plus_picture)
                pixels_lst.clear()
                label_plus_picture.clear()
                self.set_number += 1
train = Data('train-images','train-labels',1500000) #Обучающий сет, 1500000 - примерно 2000 итераций
test = Data('test-images','test-labels') #Проверочный сет
#Инициализируем веса, смещения, сигму
first_weights = np.random.rand(112,784)/1000
first_bias = np.random.rand(112)
second_weights = np.random.rand(10,112)/10
second_bias = np.random.rand(10)
def sigma(summ,bias):
    return 1/(1+np.exp(-summ + bias))
def dsigma(summ,bias):
    return 1/(1+np.exp(-summ + bias)) - (1/(1+np.exp(-summ + bias)))**2
#Один раз проходим по всем данным и считаем среднюю ошибку
average_loss = [0]
for i in train.data_set:
    correct_answer = np.zeros(10)
    correct_answer[train.data_set[i][0]] = 1
    input_layer = np.array(train.data_set[i][1],float)
    hiden_layer_sigma = sigma(np.dot(first_weights,input_layer),first_bias)
    output_layer_sigma = sigma(np.dot(second_weights,hiden_layer_sigma),second_bias)
    output_loss = output_layer_sigma - correct_answer
    for j in output_loss:
        average_loss[0] += (1/10)*(j)**2
average_loss[0] /= train.set_number+1
#Учим сеть
for i in train.data_set:
    print ('ЦИКЛ ОБУЧЕНИЯ', i+1, 'ИЗ', train.set_number+1)
    average_loss.append(average_loss[-1])
    stat = average_loss[-1]
    count = 0
    while True:
        #Прямой ход
        loss_func = 0
        correct_answer = np.zeros(10)
        correct_answer[train.data_set[i][0]] = 1
        input_layer = np.array(train.data_set[i][1],float)
        hiden_layer = np.dot(first_weights,input_layer)
        hiden_layer_sigma = sigma(hiden_layer,first_bias)
        output_layer = np.dot(second_weights,hiden_layer_sigma)
        output_layer_sigma = sigma(output_layer,second_bias)
        output_loss = output_layer_sigma - correct_answer
        for j in output_loss:
            loss_func += (1/10)*(j)**2
        average_loss[-1] = 0.9*average_loss[-1] + 0.1*loss_func
        if count == 10:
            if abs(stat - average_loss[-1]) <= 0.01:
                break
            count = 0
            if stat > average_loss[-1]:
                stat = average_loss[-1]
        #обратный ход
        step = 0.0001
        hiden_layer_loss = np.dot(np.outer(output_loss, dsigma(output_layer,second_bias)).diagonal(),second_weights)
        first_bias_delta = np.outer(hiden_layer_loss, dsigma(hiden_layer,first_bias)).diagonal()
        second_bias_delta = np.outer(output_loss, dsigma(output_layer,second_bias)).diagonal()
        second_weights -= step*np.outer(np.outer(output_loss, dsigma(output_layer,second_bias)).diagonal(),hiden_layer)
        first_weights -= step*np.outer(np.outer(hiden_layer_loss, dsigma(hiden_layer,first_bias)).diagonal(),input_layer)
        first_bias -= step*first_bias_delta
        second_bias -= step*second_bias_delta
        count += 1
#Проверяем
correct_answer_count = 0
for i in test.data_set:
    print ('Распознаю картинку номер',i+1)
    correct_answer = np.zeros(10)
    correct_answer[test.data_set[i][0]] = 1
    input_layer = np.array(test.data_set[i][1],float)
    hiden_layer_sigma = sigma(np.dot(first_weights,input_layer), first_bias)
    output_layer_sigma = sigma(np.dot(second_weights,hiden_layer_sigma),second_bias)
    if correct_answer.argmax()== output_layer_sigma.argmax():
        correct_answer_count += 1
print ('Правильных ответов', correct_answer_count, 'из', test.set_number, 'или',(correct_answer_count*100)/(test.set_number),'%' )
average_loss.pop(0)
plt.figure()
plt.scatter(np.array([j for j in range(train.set_number)]),np.array(average_loss))
plt.ylabel('Loss func')
plt.xlabel('Iter')
plt.show()