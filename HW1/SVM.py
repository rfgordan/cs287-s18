A
# coding: utf-8

# In[82]:

# Text text processing library and methods for pretrained word embeddings
import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sklearn import svm
import sklearn


# In[96]:

# Our input $x$
TEXT = torchtext.data.Field()

# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=-1, repeat = False)


# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


# In[13]:

e=nn.Embedding(len(TEXT.vocab),TEXT.vocab.vectors.size(1))
e.weight.data = TEXT.vocab.vectors


# In[97]:

def extract_matrix(gen):
    texts = []
    labels = []
    for batch in gen:
        embedded = e(batch.text)
        pooled = torch.sum(embedded,0)
        split = torch.chunk(pooled,10,0)
        numped = [torch.squeeze(ele).data.numpy() for ele in split]
        texts += numped
        labels += [torch.squeeze(ele).data for ele in torch.chunk(batch.label - 1,10,0)]
        
    return (texts,labels)


args = extract_matrix(train_iter)


# In[98]:

clf = svm.SVC()
clf.fit(*args)


# In[99]:

val_data =  extract_matrix(val_iter)

y_pred = clf.predict(val_data[0])

print(sklearn.metrics.accuracy_score(val_data[1],y_pred))


# In[95]:




# In[ ]:



