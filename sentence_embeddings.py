import numpy as np
import nltk
import os, sys
from scipy import stats
# import helper
# import seaborn as sns
import matplotlib.pyplot as plt

# config, PROJECT_ROOT = helper.get_config()
gloveFile = "/home/user/glove_pretrained_models/glove.6B.50d.txt"
# sys.path.append(".")

def load_pretrained_word2vec_model():
    f2 = open(gloveFile, 'r')
    model = {}
    for line in f2:
        splitline = line.split()
        word = splitline[0]
        embedding = [float(val) for val in splitline[1:]]
        model[word] = embedding
    return model



def get_features_for_sent(sentence, pre_trained_word2vec,max_len_sen ):

    sentence = sentence.split(" ")
    sentence_vecs = []
    each_vec_len=0
    for each_word in sentence:
        try:
            word_vec = pre_trained_word2vec[each_word]
        except:
            word_vec = pre_trained_word2vec["a"]
        each_vec_len=len(word_vec)
        sentence_vecs.extend(word_vec)

    if max_len_sen!=0:
        for j in range(len(sentence),max_len_sen):
            sentence_vecs.extend([0] * each_vec_len)
        return sentence_vecs

pretrained = load_pretrained_word2vec_model()

v1 = get_features_for_sent("this is a sentence", pretrained, 62)
v2 = get_features_for_sent("second sentence", pretrained, 62)
v3 = get_features_for_sent("third one", pretrained, 62)

import matplotlib.pyplot as plt

M = np.array([v1, v2, v3])


print(M)
rows,cols = M.T.shape

#Get absolute maxes for axis ranges to center origin
#This is optional
maxes = 1.1*np.amax(abs(M), axis = 0)

for i,l in enumerate(range(0,cols)):
    xs = [0,M[i,0]]
    ys = [0,M[i,1]]
    plt.plot(xs,ys)

plt.plot(0,0,'ok') #<-- plot a black point at the origin
plt.axis('equal')  #<-- set the axes to the same scale
plt.xlim([-maxes[0],maxes[0]]) #<-- set the x axis limits
plt.ylim([-maxes[1],maxes[1]]) #<-- set the y axis limits
plt.legend(['V'+str(i+1) for i in range(cols)]) #<-- give a legend
plt.grid(b=True, which='major') #<-- plot grid lines
plt.show()
#
#


