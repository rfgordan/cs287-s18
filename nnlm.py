import torch
import math
import numpy as np
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class NNLM(nn.Module):
    def __init__(self, hidden_size, nlayers=1, maxnorm = 1, device=-1):
        super(NNLM, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(len(TEXT.vocab), hidden_size, maxnorm)
        self.conv = nn.Conv1d(hidden_size, hidden_size, 2)
        #self.training=True
        layers = []
        for i in range(nlayers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
            #layers.append(nn.Dropout(dropout, training=self.training))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, len(TEXT.vocab))
        self.device = device

    def forward(self, text):
        x = self.embed(text)
        padding = Variable(x.data.new(1, x.size()[1], x.size()[2]))
        padding.requires_grad = False
        padding.data.fill_(0)
        x = torch.cat([padding, x], 0)
        y = self.conv(x.permute(1,2,0)).permute(2,0,1)
        return self.fc(self.layers(y)).view(-1, len(TEXT.vocab))

    def train(self):
        CEL_w = torch.FloatTensor(np.ones(len(TEXT.vocab)))
        CEL_w[TEXT.vocab.stoi["<pad>"]] = 0
        if self.device >= 0:
            CEL_w = CEL_w.cuda()
        loss_fn = nn.CrossEntropyLoss(weight=Variable(CEL_w), size_average=False)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params)
        train_loss = 0
        total_t_words = 0
        valid_loss = 0
        total_v_words = 0
        best_valid_loss = 200000
        for epoch in range(50):
            tic = time.time()
            for batch in train_iter:
                #self.training=True
                with torch.cuda.device(0):
                    optimizer.zero_grad()
                    probs= self.forward(batch.text)
                    # print(probs)
                    # print(batch.target.view(-1))
                    loss = loss_fn(probs, batch.target.view(-1))
                    loss.backward()
                    train_loss += loss
                    total_t_words += batch.target.ne(TEXT.vocab.stoi["<pad>"]).int().sum()
                    nn.utils.clip_grad_norm(self.parameters(), 2)
                    optimizer.step()
            for batch in valid_iter:
                #self.training=False
                with torch.cuda.device(0):
                    probs = self.forward(batch.text)
                    loss = loss_fn(probs, batch.target.view(-1))
                    valid_loss += loss
                    total_v_words += batch.target.ne(TEXT.vocab.stoi["<pad>"]).int().sum()
            print('[epoch: {:d}], train_loss: {:.3f}, valid_loss: {:.3f}, ({:.1f}s)'.format(epoch, math.exp(train_loss.data[0] / total_t_words.data[0]),math.exp(valid_loss.data[0] / total_v_words.data[0]), time.time()-tic) )
            if valid_loss.data[0] <= best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model, "NNLM.pth")
            #print('[epoch: {:d}], train_loss: {:.3f},({:.1f}s)'.format(epoch, math.exp(train_loss.data[0] / total_t_words.data[0]), time.time()-tic) )
path=False
TEXT = torchtext.data.Field()
torch.backends.cudnn.enabled = False
device = 1
TEXT = torchtext.data.Field()
train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
(train, valid, test), batch_size=128, bptt_len=32, device = 0, repeat=False)
if path:
    model.load('NNLM.pth')
else:
    model = NNLM(128,  nlayers=2, device=device)
if device == 1:
    model.cuda()
model.train()
