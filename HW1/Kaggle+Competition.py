
# coding: utf-8

# In[ ]:

# Text text processing library and methods for pretrained word embeddings
import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt


# In[47]:

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


# In[172]:


def print_valid(model, loss_function):
    #track loss over several validation batches
    val_loss = 0
    num_val_batches = 0


    correct = 0.0
    pred_pos = 0
    pred_neg = 0

    model.eval()
    
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

    #print(val_loss/num_val_batches)


    model.train()
    
    total = num_val_batches*10.0

    print("Validation Accuracy: %f, Positive predictions: %f, Negative Predictions: %f, Correct: %f" % (correct/total, pred_pos,pred_neg,correct))
    
    return correct/total


# In[151]:

#LSTM that predicts based on last hidden state
class LCA_LSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        
        super(LCA_LSTM,self).__init__()
        
        self.hidden_dim = hidden_dim
        
        #define layers of our model
        self.embedding = nn.Embedding(vocab_size, TEXT.vocab.vectors.size()[1])
        self.embedding.weight.data = TEXT.vocab.vectors
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(TEXT.vocab.vectors.size()[1], hidden_dim)
        self.hidden = self.init_hidden()
        self.final = nn.Linear(hidden_dim, 2)
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        
    def init_hidden(self):
        return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))
    
    def forward(self, sentence):
        
        #get embedded vectors
        vectors = self.embedding(sentence)
        
        vectors = self.d1(vectors)
        
        #clean hidden layer
        self.hidden = self.init_hidden()
        
        #pass through the entire sentence
        seq, self.hidden = self.lstm(vectors)
        
        #activate? is this necessary...experiment
        #activated = F.relu(seq[-1])
        
        #pool features from every hidden state?
        pooled = torch.sum(seq,0)
        
        #map to binary
        out = self.final(pooled)
        
        return F.log_softmax(out,dim=1)
        
        


# In[152]:

def train_LSTM():
    
    model = LCA_LSTM(5, 4, len(TEXT.vocab))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    
    
    
    epochs = 200
    for epoch in range(epochs):
        
        #tracks batches (allows us to have repeats in train_iter)
        num_batches = 0
        
        correct = 0.0
        pred_pos = 0
        pred_neg = 0
        
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
            
            _, argmax = log_probs.max(1)
            pred_pos += (argmax==0).sum().data[0]
            pred_neg += (argmax==1).sum().data[0]
            correct += (argmax==batch.label-1).sum().data[0]
            
            #halt if made on full pass
            if num_batches >= len(train):
                break
            
            
        if epoch % 5 == 0:
            
            print("Epoch: %f" % epoch)
            
            #check performance on validation data 
            print_valid(model, loss_function)
            
            #print performance on training data
            print("Training Accuracy: %f Positive predictions: %f, Negative Predictions: %f, Correct: %f" % (correct/num_batches, pred_pos,pred_neg,correct))
    return model

my_model = train_LSTM()


# In[161]:

#LSTM that predicts based on last hidden state
class LCA_BI_LSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        
        super(LCA_BI_LSTM,self).__init__()
        
        self.hidden_dim = hidden_dim
        
        embedding_size = TEXT.vocab.vectors.size()[1]
        
        #define layers of our model
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = TEXT.vocab.vectors[:,:embedding_size]
        self.embedding.weight.requires_grad = True
        self.lstm1 = nn.LSTM(embedding_size, hidden_dim, dropout=1)
        self.lstm2 = nn.LSTM(embedding_size, hidden_dim, dropout=1)
        self.hidden = self.init_hidden()
        self.final = nn.Linear(self.hidden_dim*2, 2)
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        
    def init_hidden(self):
        return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))
    
    def forward(self, sentence):
        
        #get embedded vectors
        vectors = self.embedding(sentence)
        
        vectors = self.d1(vectors)
        #clean hidden layer
        self.hidden = self.init_hidden()
        
        #pass through the entire sentence
        seq, self.hidden = self.lstm1(vectors)
        
        #flip vectors
        inv_idx = Variable(torch.arange(vectors.size(0)-1, -1, -1).long())
        inv_idx.requires_grad = False
        rev_vectors = torch.index_select(vectors, 0, inv_idx)
        
        #pass reversed sequence to second lstm
        seq2, _ = self.lstm2(rev_vectors)
        
        #activate? is this necessary...experiment
        #activated = F.relu(seq[-1])
        
        #concat two final hidden states
        combined = torch.cat([seq[-1],seq2[-1]],1)
        
        combined = self.d2(combined)
        #map to binary
        out = self.final(combined)
        
        return F.log_softmax(out,dim=1)


# In[ ]:

def train_BI_LSTM(fine_logging=False):
    
    model = LCA_BI_LSTM(4, 3, len(TEXT.vocab))
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
    
    
    #record accuracies for plotting
    valid_accs = []
    training_accs = []
    
    epochs = 20
    for epoch in range(epochs):
        
        #tracks batches (allows us to have repeats in train_iter)
        num_batches = 0
        
        correct = 0.0
        pred_pos = 0
        pred_neg = 0
        
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
            
            _, argmax = log_probs.max(1)
            pred_pos += (argmax==0).sum().data[0]
            pred_neg += (argmax==1).sum().data[0]
            correct += (argmax==batch.label-1).sum().data[0]
            
            #log performance every 10 batches
            if fine_logging and num_batches % 10 == 0 and num_batches > 0:
                valid_accs += [print_valid(model,loss_function)]
                training_accs += [correct/10.]
                correct = 0
            
            #halt if made on full pass
            if num_batches >= len(train):
                break
            
            
        if epoch % 5 == 0 and fine_logging is False :
            
            print("Epoch: %f" % epoch)
            
            #check performance on validation data 
            valid_accs += [print_valid(model, loss_function)]
            
            #print performance on training data
            print("Training Accuracy: %f Positive predictions: %f, Negative Predictions: %f, Correct: %f" % (correct/num_batches, pred_pos,pred_neg,correct))
            training_accs += [correct/num_batches]
            
    return (model, training_accs, valid_accs)

my_model, t, v = train_BI_LSTM(fine_logging=True)


# In[102]:




# In[95]:




# In[171]:




# In[ ]:



