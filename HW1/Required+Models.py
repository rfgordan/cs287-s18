
# coding: utf-8

# In[26]:

# Text text processing library and methods for pretrained word embeddings
import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[201]:

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
    (train, val, test), batch_size=10, device=-1)

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


# In[167]:

len(train)


# In[ ]:

class LCA_CNN(nn.Module):
    def __init__(self, num_labels, vocab_size):
        
    def forward(self, text_vec):
        


# In[237]:

#create our class of Naive Bayes
class LCA_LR(nn.Module):
    
    #create vectors for probs given each feature
    def __init__(self, num_labels, vocab_size):
        super(LCA_LR, self).__init__()
        self.weights = nn.Embedding(vocab_size, num_labels)
        self.bias = Variable(torch.FloatTensor([0]))
        
    #forward pass given bow vector
    def forward(self, text_vec):
#        print(text_vec.size())
        w = self.weights(text_vec)
#        print(w.size())
        summed = torch.sum(w,dim=0) + self.bias
#        print(summed.size())
        return F.log_softmax(summed,dim=1)


# In[238]:

class LCA_CBOW(nn.Module):
    
    def __init__(self, num_labels, vocab_size, proj1,proj2):
        super(LCA_CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, TEXT.vocab.vectors.size()[1])
        self.embedding.weight.data = TEXT.vocab.vectors
        self.h1 = nn.Linear(TEXT.vocab.vectors.size()[1], proj2)
        self.h2 = nn.Linear(proj2, num_labels)
        
        
    def forward(self, text_vec):
        w = self.embedding(text_vec)
        summed = torch.sum(w,dim=0)
        w2 = self.h1(summed)
        a1 = F.relu(w2)
        w3 = self.h2(a1)
        
        return F.log_softmax(w2,dim=1)
        


# In[239]:

def train_CBOW(proj1,proj2):
    model = LCA_CBOW(len(LABEL.vocab)-1,len(TEXT.vocab),proj1,proj2)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    epochs = 50
    for epoch in range(epochs):
        
        #tracks batches (allows us to have repeats in train_iter)
        num_batches = 0
        
        for batch in train_iter:
        
        
        #print(text_vec)
            
            model.zero_grad()
            
            #add in a bias term
            text_vec = batch.text
            
            log_probs = model(text_vec)

            loss = loss_function(log_probs, batch.label - 1)
            loss.backward()
            optimizer.step()
        
            num_batches+=batch.label.size()[0]
            
            #halt if made on full pass
            if num_batches >= len(train):
                break
            
            
        if epoch % 5 == 0:
            #track loss over several validation batches
            val_loss = 0
            num_val_batches = 0


            correct = 0.0
            pred_pos = 0
            pred_neg = 0
            pred_neut = 0

            #loop over and compute loss for every validation batch
            for batch in val_iter:


            #print(text_vec)


                log_probs = model(batch.text)


                _, argmax = log_probs.max(1)
                pred_pos += (argmax==0).sum().data[0]
                pred_neg += (argmax==1).sum().data[0]
                correct += (argmax==batch.label-1).sum().data[0]
                loss = loss_function(log_probs, batch.label - 1)

                val_loss += loss
                num_val_batches += 1

                if num_val_batches*10 >= len(val):
                    break

            print(val_loss/num_val_batches)


            total = num_val_batches*10.0

            print("Validation Accuracy: %f, Neutral prediction: %f, Positive predictions: %f, Negative Predictions: %f, Correct: %f" % (correct/total,pred_neut, pred_pos,pred_neg,correct))

    return model

my_model = train_CBOW(10,5)


# In[240]:


#create and train logistic regression model
def train_LR(model = None):
    
    if not model:
        model = LCA_LR(len(LABEL.vocab)-1,len(TEXT.vocab))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 50
    for epoch in range(epochs):

        #tracks batches (allows us to have repeats in train_iter)
        num_batches = 0

        for batch in train_iter:


            #print(text_vec)

            model.zero_grad()

            #add in a bias term
            #text_vec = torch.cat((batch.text,Variable(torch.LongTensor(1,10).fill_(len(TEXT.vocab)))),0)

            log_probs = model(batch.text)

            loss = loss_function(log_probs, batch.label-1)
            loss.backward()
            optimizer.step()

            num_batches+=batch.label.size()[0]

            #halt if made on full pass
            if num_batches >= len(train):
                break


        if epoch % 5 == 0:

            #track loss over several validation batches
            val_loss = 0
            num_val_batches = 0


            correct = 0.0
            pred_pos = 0
            pred_neg = 0
            pred_neut = 0

            #loop over and compute loss for every validation batch
            for batch in val_iter:


            #print(text_vec)


                #text_vec = torch.cat((batch.text,Variable(torch.LongTensor(1,batch.text.size()[1]).fill_(len(TEXT.vocab)))),0)

                log_probs = model(batch.text)


                _, argmax = log_probs.max(1)
                pred_pos += (argmax==0).sum().data[0]
                pred_neg += (argmax==1).sum().data[0]
                correct += (argmax==batch.label-1).sum().data[0]
                loss = loss_function(log_probs, batch.label-1)

                val_loss += loss
                num_val_batches += 1

                if num_val_batches*10 >= len(val):
                    break

            print(val_loss/num_val_batches)


            total = num_val_batches*10.0

            print("Validation Accuracy: %f, Neutral prediction: %f, Positive predictions: %f, Negative Predictions: %f, Correct: %f" % (correct/total,pred_neut, pred_pos,pred_neg,correct))

    return model

my_model = train_LR()    


# In[116]:

def test(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        _, argmax = probs.max(1)
        upload += list(argmax.data)

    with open("predictions.txt", "w") as f:
        for u in upload:
            f.write(str(u) + "\n")


# In[114]:

def valid(model):
    batch_size = 10
    valid_iter = torchtext.data.BucketIterator(val, train=False, batch_size=10,repeat = False)
    correct = 0
    total = 0
    pred_pos = 0
    pred_neg = 0
    for batch in valid_iter:
        probs = model(batch.text)
        _, argmax = probs.max(1)
        pred_pos += (argmax==1).sum()
        pred_neg += (argmax==2).sum()
        correct += (argmax==batch.label).sum()
        total += batch_size
        
    print("Validation Accuracy: %f, Positive predictions: %f, Negative Predictions: %f", correct/float(total),pred_pos,pred_neg)


# In[ ]:



