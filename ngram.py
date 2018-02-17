# Text text processing library and methods for pretrained word embeddings
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

class Trigram_model(nn.Module):
    def __init__(self, text,dev_text,device=-1):
        super(Trigram_model, self).__init__()
        self.counts = {}
        self.dev_counts = {}
        self.train_text= text
        self.dev_text = dev_text
        self.gamma = 0.1
        self.lambdas = [0, 0, 1]

    def q(self, prev, w):
        if len(prev) == 0 and tuple(w) in self.counts:
            return self.counts[tuple(w)] / len(self.counts)
        elif tuple(prev) in self.counts and tuple(prev + w) in self.counts:
            return self.counts[tuple(prev + w)]/ self.counts[tuple(prev)]
        else:
            return 0

    def forward(self, words, lambdas):
        return 0.000001+ lambdas[0] * self.q(words[:2], [words[2]]) + lambdas[1] * self.q([words[1]], [words[2]]) + lambdas[2] * self.q([], [words[2]])

    def output(self, words):
        if tuple(words) in self.counts:
            bigram = self.counts[tuple(words)]
        else:
            bigram = 0
        if tuple([words[0]]) in self.counts:
            unigram = self.counts[tuple([words[0]])]
        else:
            unigram = 0
        l1 = bigram / (bigram + self.gamma)
        l2 = (1 - l1) *  unigram/ ( unigram + self.gamma)
        l3 = 1-l1-l2
        output_v = np.zeros(len(TEXT.vocab))
        for i in range(len(TEXT.vocab)):
            output_v[i] = self.forward(words + [TEXT.vocab.itos[i]], [l1, l2, l3])
        return output_v

    def train(self):
        CEL_w = torch.FloatTensor(np.ones(len(TEXT.vocab)))
        CEL_w[TEXT.vocab.stoi["<pad>"]] = 0
        loss_fn = nn.CrossEntropyLoss(weight=Variable(CEL_w), size_average=False)
        train_loss = 0
        total_t_words = 0
        valid_loss = 0
        total_v_words = 0
        best_valid_loss = 200000
        for word in self.train_text:
            key = tuple([word])
            if key in self.counts:
                self.counts[key] += 1
            else:
                self.counts[key] = 1

        for i in range(len(self.train_text) - 1):
            key = tuple(self.train_text[i:i+2])
            if key in self.counts:
                self.counts[key] += 1
            else:
                self.counts[key] = 1

        for i in range(len(self.train_text) - 2):
            key = tuple(self.train_text[i:i+3])
            if key in self.counts:
                self.counts[key] += 1
            else:
                self.counts[key] = 1


        best_n_log_lik = 10 ** 10
        best_gamma = 0
        for gamma in np.arange(0.1, 5, 0.5):
            n_log_lik = 0
            for i in range(len(self.dev_text) - 2):
                words = self.dev_text[i:i+3]
                #print(words)
                if tuple(words[:2]) in self.counts:
                    bigram = self.counts[tuple(words[:2])]
                else:
                    bigram = 0

                if tuple([words[0]]) in self.counts:
                    unigram = self.counts[tuple([words[0]])]
                else:
                    unigram = 0
                if bigram in self.dev_counts:
                    self.dev_counts[bigram] += 1
                else:
                    self.dev_counts[bigram] = 1
                l1 = bigram / (bigram + gamma)
                l2 = (1 - l1) *  unigram / ( unigram+ gamma)
                l3 = 1-l1-l2
                #print(self.forward(words, [l1, l2, l3]))
                n_log_lik += -1*self.dev_counts[bigram] * np.log(self.forward(words, [l1, l2, l3]))
            if n_log_lik <= best_n_log_lik:
                best_n_log_lik = n_log_lik
                best_gamma = gamma
            print(gamma, n_log_lik)
        print('Best gamma', best_gamma)
        print('Best n_log_lik', best_n_log_lik)
        self.gamma = best_gamma
        for batch in valid_iter:
            with torch.cuda.device(0):
                last_rows = torch.t(batch.text)
                answer = np.zeros((len(last_rows),len(last_rows[0]) ,len(TEXT.vocab)))
                for i in range(len(last_rows)):
                    print(i)
                    answer[i][0] = self.output(['Mingu', TEXT.vocab.itos[last_rows[i][0].data[0]]])
                    for j in range(len(last_rows[0]) - 1):
                        answer[i][j+1] = self.output([TEXT.vocab.itos[word] for word in last_rows[i][j:j+2].data])
                loss = loss_fn(Variable(torch.FloatTensor(answer)).view(-1, len(TEXT.vocab)), batch.target.view(-1))
                valid_loss += loss
                total_v_words += batch.target.ne(TEXT.vocab.stoi["<pad>"]).int().sum()
                print('valid_loss: {:.3f}'.format(math.exp(valid_loss.data[0] / total_v_words.data[0])) )

TEXT = torchtext.data.Field()
torch.backends.cudnn.enabled = False
device = 1
TEXT = torchtext.data.Field()
train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
(train, valid, test), batch_size=128, bptt_len=32, device = -1, repeat=False)
train_text = next(train.text)
model = Trigram_model(train_text[:int(0.9*len(train_text))], train_text[int(0.9*len(train_text)):])
if device == 1:
    model.cuda()
model.train()
