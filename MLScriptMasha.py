
# coding: utf-8

# In[17]:


import pandas as pd
import sklearn
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import numpy as np
#from nltk.corpus import stopwords
#stops = set(stopwords.words("english"))



#import training and test data
testText = pd.read_csv('test_text', sep="\|\|", engine='python', header=None, names=["ID","Text"])
testVariants = pd.read_csv('test_variants', sep=",", engine = "python", header=None, names=["ID","Gene","Variation","Class"])
trainingText = pd.read_csv('training_text', sep="\|\|", engine='python', skiprows=1, names=["ID","Text"])
trainingVariants= pd.read_csv('training_variants', sep=",", engine="python")

#define x and y variables 
trainingX = trainingText
trainingY = trainingVariants.drop(['Class'], axis=1)
trainingX, trainingY = shuffle(trainingX, trainingY)
print("before cleaning:", len(trainingX))

trainingX = trainingX[trainingX['Text'] != 'null']
trainingX['Text'] = trainingX['Text'].str.replace(r"[^A-Za-z0-9^,!.\/'+-=]", ' ')
trainingX['Text'] = trainingX['Text'].str.lower()
trainingX['Text'] = trainingX['Text'].str.split()
#trainingX['Text'] = trainingX['Text'].apply(lambda x: [w for w in x if not w in stops])  
#trainingX['Text']= trainingX['Text'].str.translate(str.maketrans("","", string.punctuation))
print("after cleaning:", len(trainingX))
trainingY = trainingY[trainingY.ID.isin(trainingX['ID'])]


testingX = testText
testingY = testVariants.drop(['Class'], axis=1)
testingX, testingY = shuffle(testingX, testingY)

testingX = testingX[testingX['Text'] != 'null']
testingX['Text'] = testingX['Text'].str.replace(r"[^A-Za-z0-9^,!.\/'+-=]", ' ')
testingX['Text'] = testingX['Text'].str.lower()
testingX['Text'] = testingX['Text'].str.split()
#trainingX['Text']= trainingX['Text'].str.translate(str.maketrans("","", string.punctuation))
testingY = testingY[testingY.ID.isin(testingX['ID'])]



# In[7]:


# trainingX.head()
# trainingY.head()
# testingMerged = pd.concat([trainingX,trainingY], axis=1)
# testingMerged.head()


# In[18]:


from gensim.models.doc2vec import TaggedDocument
from gensim import utils

excerpts=[]
for index, row in trainingX['Text'].iteritems():
    concatText = " ".join(row)
    excerpts.append(TaggedDocument(utils.to_unicode(concatText).split(), ['Text' + '_%s' % str(index)]))

for index, row in testingX['Text'].iteritems():
    concatText = " ".join(row)
    excerpts.append(TaggedDocument(utils.to_unicode(concatText).split(), ['Text' + '_%s' % str(index)]))


# In[39]:


from gensim.models import Doc2Vec
import os

Text_INPUT_DIM=50
filename='preprocessedText50.d2v'

# if os.path.isfile(filename):
#     text_model = Doc2Vec.load(filename)
    
# else:

text_model = Doc2Vec(min_count=1, window=5, size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
text_model.build_vocab(excerpts)
text_model.train(excerpts, total_examples=text_model.corpus_count, epochs=text_model.iter)
text_model.save(filename)




# In[40]:


#print(text_model)

train_size= len(trainingX)
test_size = len(testingX)


text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))
text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))

for i, val in enumerate(trainingX['ID']):
    text_train_arrays[i] = text_model.docvecs['Text_'+str(val)]

for j, val2 in enumerate(testingX['ID']):
    text_test_arrays[j] = text_model.docvecs['Text_'+str(val2)]

print(text_train_arrays[0][:50])


# In[33]:


#print(trainingY.groupby(['Gene','Variation']).size().reset_index().rename(columns={0:'count'}))


# In[41]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD

def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_dim=50, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dense(267, init='normal', activation="softmax"))
    
#     model.add(Dense(256, input_dim=Text_INPUT_DIM, init='normal', activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(256, init='normal', activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(80, init='normal', activation='relu'))
#     model.add(Dense(262, init='normal', activation="softmax"))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[42]:


model = baseline_model()
model.summary()


# In[43]:


#264 unique genes


# In[44]:


one_hot_gene = pd.get_dummies(trainingY['Gene'])
one_hot_gene_test = pd.get_dummies(testingY['Gene'])
missing_cols = set(one_hot_gene.columns) - set(one_hot_gene_test.columns )
for c in missing_cols:
    one_hot_gene_test[c] = 0

missing_cols2 = set(one_hot_gene_test.columns) - set(one_hot_gene.columns )
for c in missing_cols2:
    one_hot_gene[c] = 0
    
one_hot_gene_test = one_hot_gene_test[one_hot_gene.columns]

estimator=model.fit(text_train_arrays, one_hot_gene, validation_split=0.5, epochs=10, batch_size=64)

print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))


# In[45]:



results = model.evaluate(text_test_arrays, one_hot_gene_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
        
print(model.metrics_names, results)


# In[47]:


#plotting the text length distribution: violin plot
# plt.figure(figsize=(12,8))

# gene_count_grp = trainingSet["Text_count"].sum()
# sns.violinplot(y="Text_count", data=trainingSet, inner=None)
# sns.swarmplot(y="Text_count", data=trainingSet, color="w", alpha=.5);
# plt.ylabel('Text Count', fontsize=14)
# plt.xlabel('Training Dataset', fontsize=14)
# plt.title("Text length distribution", fontsize=18)
# plt.show()

plt.figure(figsize=(12, 8))
sns.distplot(trainingSet.Text_count.values, bins=50, kde=False, color='red')
plt.xlabel('Number of words in text', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Frequency of number of words", fontsize=15)
plt.show()


# In[51]:




