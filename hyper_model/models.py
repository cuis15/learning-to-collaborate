from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import pdb



class LocalOutput(nn.Module):
    '''
    output layer module
    '''
    def __init__(self, n_input=84, n_output=2, nonlinearity=False):
        super().__init__()
        self.nonlinearity = nonlinearity
        layers = []
        if nonlinearity:
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_input, n_output))
        layers.append(nn.Softmax(dim=1))
        self.layer = nn.Sequential(*layers)
        self.loss_CE = nn.CrossEntropyLoss()
    def forward(self, x, y):
        pred = self.layer(x)
        loss = self.loss_CE(pred, y)
        return pred, loss 



class HyperSimpleNet(nn.Module):
    '''
    hypersimplenet for adult and synthetic experiments
    '''
    def __init__(self, args, device):
        super(HyperSimpleNet, self).__init__()
        self.n_users = args.num_users
        usr_used = [i for i in range(self.n_users)]
        self.input_ray = Variable(torch.FloatTensor([[1/len(usr_used) for i in usr_used]])).to(device)
        # self.input_ray = Variable(torch.FloatTensor([[0.25,0.25,0.25,0.05,0.05,0.05]])).to(device)
        self.input_ray.requires_grad = True
        self.dataset = args.dataset
        hidden_dim = args.hidden_dim
        spec_norm = args.spec_norm
        layers = [ spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim)]
        for _ in range(args.n_hidden - 1):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.input_dim = args.input_dim

        if self.dataset == "synthetic":
            self.loss = nn.MSELoss(reduction='mean')
            self.l1_weights = nn.Linear(hidden_dim, self.input_dim * 1)
            self.l1_bias = nn.Linear(hidden_dim, 1)
            self.output_dim = 1

        elif self.dataset == "adult":
            self.layer = nn.Sigmoid()
            self.loss  = nn.BCELoss()
            self.l1_weights = nn.Linear(hidden_dim, self.input_dim * 1)
            self.l1_bias = nn.Linear(hidden_dim, 1)
            self.output_dim = 1


    def forward(self, x, y, usr_id,  input_ray = None):
        if input_ray!=None:
            self.input_ray.data = input_ray
        feature = self.mlp(self.input_ray)
        l1_weight = self.l1_weights(feature).view(self.output_dim, self.input_dim)
        l1_bias = self.l1_bias(feature).view(-1)
        x = F.linear(x, weight=l1_weight, bias=l1_bias)
        if self.dataset == "synthetic":
            pred = x.flatten()
        elif self.dataset == "adult":
            pred = self.layer(x).flatten()
        y = y.float()
        loss = self.loss(pred,y)
        return pred, loss





class Basenet(nn.Module):
    def __init__(self, args):
        super(Basenet).__init__()

        self.fc0 = nn.Linear(429, args.embedding_dim)
        self.fc1 = nn.Linear( (args.embedding+13)*200, 64)
        self.fc2 = nn.Linear( 64, 1)
        self.actv = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x, y, usr_id):
        x1 = x[:, :, :429]
        x2 = x[:, :, 429:]
        x1 = x1.view(-1, 429)
        embed_x1 = self.fc0(x1)
        embed_x1 = embed_x1.view(-1, 200, self.args.embedding_dim)
        x = torch.cat([embed_x1, x2], dim = 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        logits = self.fc2(x)
        pred = self.actv(logits).flatten()
        loss = self.loss(pred, y)
        return pred, loss 




class Hypernet(nn.Module):
    '''
    Hypernet for CIFAR10 experiments
    '''
    def __init__(
            self, n_usrs, usr_used,  device, n_classes = 10,  in_channels=3, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden = 2):
        super().__init__()
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.input_ray = Variable(torch.FloatTensor([[1/len(usr_used) for i in usr_used]])).to(device)
        self.input_ray.requires_grad = True

         # self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [ spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim)]

        for _ in range(2):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)


        self.c1_weights = []
        self.c1_bias = []
        self.c2_weights = []
        self.c2_bias = []
        self.l1_weights = []
        self.l1_bias = []
        self.l2_weights = []
        self.l2_bias = []
        for _ in  range(n_hidden-1):
            self.c1_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c1_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_weights.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_bias.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_bias.append(nn.LeakyReLU(0.2, inplace=True))   


        self.c1_weights = nn.Sequential( *(self.c1_weights + [spectral_norm(nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)] ))
        self.c1_bias = nn.Sequential( *(self.c1_bias + [spectral_norm(nn.Linear(hidden_dim, self.n_kernels)) if spec_norm else nn.Linear(hidden_dim, self.n_kernels)] )) 
        self.c2_weights = nn.Sequential( *(self.c2_weights + [spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)] )) 
        self.c2_bias = nn.Sequential( *(self.c2_bias + [spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels)) if spec_norm else nn.Linear(hidden_dim, 2 * self.n_kernels)] ))  
        self.l1_weights = nn.Sequential( *(self.l1_weights + [spectral_norm(nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)] )) 
        self.l1_bias = nn.Sequential( *(self.l1_bias + [spectral_norm(nn.Linear(hidden_dim, 120)) if spec_norm else nn.Linear(hidden_dim, 120)] )) 
        self.l2_weights = nn.Sequential( *(self.l2_weights + [spectral_norm(nn.Linear(hidden_dim, 84 * 120)) if spec_norm else nn.Linear(hidden_dim, 84 * 120)] )) 
        self.l2_bias =  nn.Sequential( *(self.l2_bias + [spectral_norm(nn.Linear(hidden_dim, 84)) if spec_norm else nn.Linear(hidden_dim, 84)] )) 



        self.locals = nn.ModuleList([LocalOutput(n_output =n_classes) for i in range(self.n_users)])
        # self.locals = nn.ModuleList([LocalOutput(n_output =n_classes) for i in range(1)])
    
    def forward(self, x, y, usr_id,  input_ray = None):
        if input_ray!=None:
            self.input_ray.data = input_ray
        feature = self.mlp(self.input_ray)
        weights = {
            "conv1.weight": self.c1_weights(feature).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(feature).view(-1),
            "conv2.weight": self.c2_weights(feature).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(feature).view(-1),
            "fc1.weight": self.l1_weights(feature).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(feature).view(-1),
            "fc2.weight": self.l2_weights(feature).view(84, 120),
            "fc2.bias": self.l2_bias(feature).view(-1),
        }
        x = F.conv2d( x, weight=weights['conv1.weight'], bias=weights['conv1.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = F.conv2d( x, weight=weights['conv2.weight'], bias=weights['conv2.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(F.linear(x, weight=weights["fc1.weight"], bias=weights["fc1.bias"]), 0.2) 
        logits = F.leaky_relu(F.linear(x, weight=weights["fc2.weight"], bias=weights["fc2.bias"]), 0.2) 

        pred, loss = self.locals[usr_id](logits, y)

        return pred, loss
