

# -*- coding: utf-8 -*-
"""

@author: Merve & Levent
"""
import pandas as pd 
import numpy as np

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#%%
e=0.4
alpha = 0.3

#%%
train_pixels = train.pixels.str.split(",").tolist()
train_pixels = pd.DataFrame(train_pixels, dtype=int)
#%%
test_pixels = test.pixels.str.split(",").tolist()
test_pixels = pd.DataFrame(test_pixels, dtype=int)
 #%%CodeBook
import random
def create_codebook(n):
  rand=[]
  a=pd.DataFrame(columns=["category","pixels"])
  s=0
  e=199 
  for i in range(10):
    rand=[random.sample(range(s,e),n)]
    s+=200
    e+=200
    for j in rand:
        a=a.append(train.iloc[j],ignore_index=True)
  return a
codebook=create_codebook(5)
#%%
codebook_pixels = codebook.pixels.str.split(",").tolist()
codebook_pixels = pd.DataFrame(codebook_pixels, dtype=float)

#%% Distance
from math import sqrt

def distance(t,c,data):
    d=0.0
    for i in range(784):
        d+=(data[i][t]-codebook_pixels[i][c])**2
    return sqrt(d)

#%% En Yakın 2 Komşu
    
def en_yakin(x,data):
    distances = pd.DataFrame(columns=['indx','distance'])
    for i in range(50):
        d = distance(x,i,data)
        distances = distances.append({'indx':i,'distance':d},ignore_index=True)
        
    distances=distances.sort_values(by=['distance'])
    distances=distances.reset_index(drop=True)
    return distances
#%%
def lvq3():
    for i in range(2000):
        mins = pd.DataFrame(columns=['indx','distance'])
        mins = en_yakin(i,train_pixels)
        m1=int(mins.indx[0])
        m2=int(mins.indx[1])
        if(train.category[i]==codebook.category[m1] and train.category[i]==codebook.category[m2]):
            for j in range(784):
                codebook_pixels[j][m1]=codebook_pixels[j][m1]+e*alpha*(train_pixels[j][i]-codebook_pixels[j][m1])
                codebook_pixels[j][m2]=codebook_pixels[j][m2]+e*alpha*(train_pixels[j][i]-codebook_pixels[j][m2])
        elif(train.category[i]==codebook.category[m1] and train.category[i]!=codebook.category[m2]):
            for j in range(784):
                codebook_pixels[j][m1]=codebook_pixels[j][m1]+alpha*(train_pixels[j][i]-codebook_pixels[j][m1])
                codebook_pixels[j][m2]=codebook_pixels[j][m2]-alpha*(train_pixels[j][i]-codebook_pixels[j][m2])         
        elif(train.category[i]!=codebook.category[m1] and train.category[i]==codebook.category[m2]):
            for j in range(784):
                codebook_pixels[j][m1]=codebook_pixels[j][m1]-alpha*(train_pixels[j][i]-codebook_pixels[j][m1])
                codebook_pixels[j][m2]=codebook_pixels[j][m2]+alpha*(train_pixels[j][i]-codebook_pixels[j][m2])         
        elif(train.category[i]!=codebook.category[m1] and train.category[i]!=codebook.category[m2]):
            for j in range(784):
                codebook_pixels[j][m1]=codebook_pixels[j][m1]-alpha*(train_pixels[j][i]-codebook_pixels[j][m1])
                codebook_pixels[j][m2]=codebook_pixels[j][m2]-alpha*(train_pixels[j][i]-codebook_pixels[j][m2])         

#%%
lvq3()

#%%Test

test_sonuc = ["","","","","","","","","",""]
for i in range(1000):
   mins = pd.DataFrame(columns=['indx','distance'])
   mins = en_yakin(i,test_pixels)
   m1=int(mins.indx[0])
   a=int(test.category[i])
   if(test.category[i] == codebook.category[m1]):
       test_sonuc[a]=test_sonuc[a]+'1'
       
   else:
       test_sonuc[a]=test_sonuc[a]+'0'



test_accuracy = pd.DataFrame(columns=['class','accuracy'])
for i in range(10):
    acc = test_sonuc[i].count('1') / len (test[i])
    test_accuracy = test_accuracy.append({'class': i , 'accuracy' : acc}, ignore_index=True)
