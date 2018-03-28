import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid as make_image_grid
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["MLP_VAE"], default="MLP_VAE")
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA')
    parser.add_argument('--train_interval', type=int, default=100,
                        help='how frequently to print training status')
    return parser.parse_args()

args = parse_args()

def visualize_mnist_vae(model,dataloader,dir, num=16):
    images,_ = iter(dataloader).next()
    images = images[0:num,:,:]
    x_in = Variable(images).cuda()
    x_out,_,_ = model(x_in)
    x_out = x_out.data
    save_image(x_out.view(num, 1, 28, 28), dir)

class MLP_Encoder(nn.Module):
    def __init__(self):
        super(MLP_Encoder, self).__init__()
        self.layer1 = nn.Linear(784,400)
        self.relu = nn.ReLU()
        self.enc_mean = nn.Linear(400, 20)
        self.enc_logsig = nn.Linear(400, 20)
    def forward(self, x):
        o1 = self.relu(self.layer1(x))
        mean, logsig = self.enc_mean(o1), self.enc_logsig(o1)
        return mean, logsig

class MLP_Decoder(nn.Module):
    def __init__(self):
        super(MLP_Decoder, self).__init__()
        self.layer1 = nn.Linear(20, 400)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(400, 784)

    def forward(self, z):
        return F.sigmoid(self.layer2(self.relu(self.layer1(z))))

class MLP_VAE(nn.Module):
    def __init__(self):
        super(MLP_VAE, self).__init__()
        self.encoder = MLP_Encoder()
        self.decoder = MLP_Decoder()

    def reparameterize(self, mean, logsig):
        if self.training:
            sigma = logsig.exp_()
            eps = Variable(sigma.data.new(sigma.size()).normal_())
            return eps.mul(sigma).add_(mean)  # Reparameterization trick
        else:
            return mean

    def forward(self, x):
        mean, logsig = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mean, logsig)
        return self.decoder(z), mean, logsig

def loss_function(gen_data, real_data, mean, logsig):
    BCE = F.binary_cross_entropy(gen_data, real_data.view(-1, 784), size_average = False)
    # Given from Appendix B from VAE paper:
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    sigma = torch.exp(logsig)
    sigma_sq = sigma * sigma
    KL = -0.5 * torch.sum(1 + sigma_sq - (mean ** 2) - torch.exp(sigma_sq))
    return BCE + KL

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        gen_batch, mean, logsig = model(data)
        loss = loss_function(gen_batch, data, mean, logsig)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.train_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader) * len(data)) ))

def validate(epoch):
    model.eval()
    valid_loss = 0
    for i, (data, _) in enumerate(val_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        gen_batch, mean, logsig = model(data)
        loss = loss_function(gen_batch, data, mean, logsig)
        valid_loss += loss.data[0]
    print('Epoch: {} Validation loss: {:.4f}'.format(
          epoch, valid_loss / (len(val_loader) * len(data)) ))
    return valid_loss

if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())
    torch.manual_seed(3435)
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:-10000]
    train_label = train_label[:-10000]
    train_set = torch.utils.data.TensorDataset(train_img, train_label)
    val_set = torch.utils.data.TensorDataset(val_img, val_label)
    test_set = torch.utils.data.TensorDataset(test_img, test_label)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    models = {model.__name__: model for model in [MLP_VAE]}
    #model = models[args.model]
    model = MLP_VAE()
    if args.cuda:
        torch.backends.cudnn.enabled = False
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid_loss = validate(epoch)
        if not best_val_loss or valid_loss < best_val_loss:
            print("[!] saving model...")
            torch.save(model.state_dict(), 'saves/' + args.model + '_%d.pt' % (epoch))
            sample = Variable(torch.randn(64, 20))
            if args.cuda:
                sample = sample.cuda()
            gen_batch = model.decoder(sample)
            sample = F.sigmoid(gen_batch).cpu()
            save_image(sample.data.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png', normalize=True)
            best_val_loss = valid_loss
            #visualize_mnist_vae(model, test_loader,'results/sample_' + str(epoch) + '.png')
