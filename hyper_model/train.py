import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from tqdm import trange
import pdb
from torch.utils.data import DataLoader, Dataset
from hyper_model.models import Hypernet, HyperSimpleNet 
from hyper_model.solvers import EPOSolver, LinearScalarizationSolver
from torch.autograd import Variable
import os
from sklearn.metrics import roc_auc_score

"""
training a personalized model for each client;
"""

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Training_all(object):
    def __init__(self, args, logger, dataset_train=None, dataset_test=None, dict_user_train=None, dict_user_test=None, users_used=None):
        self.device = torch.device('cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() and args.gpus != -1 else 'cpu')
        self.args = args
        self.total_epoch = args.total_epoch
        self.epochs_per_valid = args.epochs_per_valid
        self.target_usr = args.target_usr
        if users_used == None:
            users_used = [i for i in range(self.args.num_users)]
        self.users_used = users_used
        if args.dataset == "adult" or args.dataset == "synthetic":
            self.hnet = HyperSimpleNet( args, self.device)
        elif args.dataset == "eicu":
            if args.train_baseline:
                self.hnet = Basenet(args)
            else:
                self.hnet = Hypereicu(args = args, usr_used = self.users_used, device = self.device)
        else:
            self.hnet = Hypernet(n_usrs = args.num_users, device = self.device, n_classes = args.n_classes, usr_used = self.users_used, n_hidden = args.n_hidden, spec_norm=args.spec_norm)
        self.hnet.to(self.device)
        self.optim = torch.optim.SGD(self.hnet.parameters(), lr=self.args.lr, momentum=0.5)
        dict_user_valid = set(random.sample(set(dict_user_train[self.target_usr]), int(len(dict_user_train[self.target_usr])/6)))
        dict_user_train[self.target_usr] = set(dict_user_train[self.target_usr]) - dict_user_valid
        self.data_valid = DataLoader(DatasetSplit(dataset_train, dict_user_valid), batch_size=len(dict_user_valid), shuffle=False)
        self.dataset_test = dataset_test
        self.dict_user_test = dict_user_test
        self.dataset_train = dataset_train
        self.data_test = dataset_test
        self.dict_user_train = dict_user_train
        if args.solver_type == "epo":
            self.solver = EPOSolver(len(self.users_used))
        elif args.solver_type == "linear":
            self.solver = LinearScalarizationSolver(len(self.users_used))
        if self.args.local_bs == -1:
            self.train_loaders = [enumerate(DataLoader(DatasetSplit(dataset_train, dict_user_train[idxs]), batch_size = len(dict_user_train[idxs]), shuffle=True, num_workers = args.num_workers)) for idxs in range(self.args.num_users)]
        else:
            self.train_loaders = [enumerate(DataLoader(DatasetSplit(dataset_train, dict_user_train[idxs]), batch_size = args.local_bs, shuffle=True, num_workers = args.num_workers)) for idxs in range(self.args.num_users)]
        self.logger = logger
        self.global_epoch = 0
        self.pickle_record = {"train":{}, "valid":{}}
        self.all_args_save(args)


    def all_args_save(self, args):
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent = 2)


    def train_input(self, usr_id):
        try:
            _, (X, Y) = self.train_loaders[usr_id].__next__()
        except StopIteration:
            if self.args.local_bs == -1:
            #     self.train_loaders[usr_id] = enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[self.users_used[usr_id]]), batch_size=len(self.dict_user_train[self.users_used[usr_id]]), shuffle=True, num_workers = self.args.num_workers))
            # else:
            #     self.train_loaders[usr_id] = enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[self.users_used[usr_id]]), batch_size=self.args.local_bs, shuffle=True, num_workers = self.args.num_workers))
        
                self.train_loaders[usr_id] = enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]), batch_size=len(self.dict_user_train[usr_id]), shuffle=True, num_workers = self.args.num_workers))
            else:
                self.train_loaders[usr_id] = enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]), batch_size=self.args.local_bs, shuffle=True, num_workers = self.args.num_workers))
            _, (X, Y) = self.train_loaders[usr_id].__next__()
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y


    def ray2users(self, ray):
        tmp_users_used = []
        tmp_ray = []
        for user_id, r in enumerate(ray):
            if r/ray[self.target_usr] >= 0.7:
                tmp_users_used.append(user_id)
                tmp_ray.append(r)
        return tmp_users_used, tmp_ray  


    def acc_auc(self, prob, Y, is_training = True):
        if self.args.dataset == "adult" or self.args.dataset == "eicu":
            y_pred = prob.data>=0.5
        elif self.args.dataset == "synthetic":
            if is_training:
                return 0
            else:
                return 0,0
        else:
            y_pred = prob.data.max(1)[1]
        users_acc = torch.mean((y_pred==Y).float()).item()

        if self.args.dataset == "eicu" and is_training:
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return  users_auc

        elif is_training and self.args.dataset != "eicu":
            return users_acc

        elif self.args.dataset == "eicu" and not is_training:
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return  users_acc,  users_auc   
        else:
            return users_acc, 0



    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl  
        


    def train_pt(self, global_iter): # training the Pareto Front using training data
        start_epoch = 0
        self.pickle_record["train"][str(global_iter)]["train-pt"] = {}
        for iteration in range(start_epoch, self.args.total_hnet_epoch):
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)] = {}       
            self.hnet.train()

            losses = []
            accs = {}
            loss_items = {}

            if self.args.sample_ray:
                ray = torch.from_numpy(np.random.dirichlet([1/len(self.users_used) for i in self.users_used], 1).astype(np.float32).flatten()).to(self.device)
                ray = ray.view(1, -1)

                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id) 
                    pred, loss = self.hnet(X, Y, usr_id, ray)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

            else:
                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id)
                    pred, loss = self.hnet(X, Y, usr_id)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()


            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            # for n, para in self.hnet.named_parameters():
            #     print(para.grad)
            # pdb.set_trace()
            self.optim.step()
            kl_l_p = self.losses_r([loss.item() for loss in losses], input_ray_numpy)
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)] = {}
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)]["losses"] = loss_items
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)]["accs"] = accs
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)]["input_ray"] = input_ray_numpy
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)]["alpha"] = alphas.data.cpu().numpy()
            self.pickle_record["train"][str(global_iter)]["train-pt"][str(iteration)]["a"] = kl_l_p
            self.logger.info("train hyper network: global-epoch: {}, iteration: {}, losses: {}, input_ray: {},  a:{}, alphas:{}, accs:{}.".format(
                global_iter, iteration,  loss_items, input_ray_numpy,  kl_l_p, alphas.data.cpu().numpy(), accs))




    def train_ray(self, global_iter): ## searching the optimal model on the PF using the validation data
        start_epoch = 0
        self.pickle_record["train"][str(global_iter)]["train-ray"] = {}
        for iteration in range(start_epoch, self.args.total_ray_epoch):
            self.pickle_record["train"][str(global_iter)]["train-ray"][str(iteration)] = {}       
            self.hnet.train()

            self.valid_loader = enumerate(self.data_valid)
            _, (X, Y) = self.valid_loader.__next__()
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred, loss = self.hnet(X, Y, self.target_usr)
            acc = self.acc_auc(pred, Y)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            loss.backward()
            self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
            self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1-self.args.eps_prefer)
            self.hnet.input_ray.data = self.hnet.input_ray.data/torch.sum(self.hnet.input_ray.data)
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]
            self.pickle_record["train"][str(global_iter)]["train-ray"][str(iteration)] = {}
            self.pickle_record["train"][str(global_iter)]["train-ray"][str(iteration)]["valid_loss"] = loss.item()
            self.pickle_record["train"][str(global_iter)]["train-ray"][str(iteration)]["valid_acc"] = acc
            self.pickle_record["train"][str(global_iter)]["train-ray"][str(iteration)]["input_ray"] = input_ray_numpy
            self.logger.info("train hyper preference: global-epoch: {}, iteration: {}, valid loss: {}, valid acc: {}, input_ray: {}.".format(
               global_iter,  iteration, loss.item(), acc, input_ray_numpy))



    def train_baseline(self): ## training baselines (FedAve and Local)
        if self.args.baseline_type == "fedave":
            start_epoch = self.global_epoch
            for iteration in range(start_epoch, self.total_epoch):
                self.pickle_record["train"][str(iteration)]= {}       
                self.hnet.train()

                losses = []
                accs = {}
                loss_items = {}

                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id)
                    pred, loss = self.hnet(X, Y, usr_id)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

                loss = torch.mean(torch.stack(losses))
                self.optim.zero_grad()
                loss.backward()
                # for n, para in self.hnet.named_parameters():
                #     print(para.grad)
                # pdb.set_trace()
                self.optim.step()
                self.pickle_record["train"][str(iteration)] = {}
                self.pickle_record["train"][str(iteration)]["losses"] = loss_items
                self.pickle_record["train"][str(iteration)]["accs"] = accs
                self.logger.info("train fedave network: iteration: {}, losses: {}, accs:{}.".format( iteration,  loss_items, accs))


                self.global_epoch+=1
                if (self.global_epoch+1)%self.epochs_per_valid == 0:
                    self.valid()
            

        elif self.args.baseline_type == "local":
            usr_id = self.args.target_usr
            for iteration in range(0, self.total_epoch):
                self.pickle_record["train"][str(iteration)]= {}       
                self.hnet.train()
                loss_items = {}

                X, Y = self.train_input(usr_id)
                pred, loss = self.hnet(X, Y, usr_id)
                acc = self.acc_auc(pred, Y)
                loss_items[str(usr_id)] = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.pickle_record["train"][str(iteration)] = {}
                self.pickle_record["train"][str(iteration)]["losses"] = loss_items
                self.pickle_record["train"][str(iteration)]["acc"] = acc
                self.logger.info("train local network: iteration: {}, loss: {}, acc:{}.".format(
                     iteration,  loss_items, acc))

                self.global_epoch+=1
                if (self.global_epoch+1)%self.epochs_per_valid == 0:
                    self.valid()            

        else:
            self.logger.info("error baseline type")
            exit()


    def train(self):
        if self.args.train_baseline:
            self.train_baseline()

        else:
            if self.args.sample_ray == True:
                if self.args.dataset == "cifar10":
                    self.args.baseline_type = "local"
                    self.total_epoch = 1000
                    self.epochs_per_valid = 1000
                    self.train_baseline()
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                for iteration in range(self.total_epoch):
                    self.pickle_record["train"][str(iteration)] = {}
                    self.train_pt(iteration)
                    self.train_ray(iteration)
                    self.global_epoch+=1

                    if (self.global_epoch+1)%self.args.epochs_per_valid == 0:
                        self.valid()
            
            else:
                if self.args.dataset == "cifar10":
                    self.args.baseline_type = "local"
                    self.total_epoch = 1000
                    self.epochs_per_valid = 1000
                    self.train_baseline()
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                for iteration in range(self.total_epoch):
                    self.pickle_record["train"][str(iteration)] = {}
                    self.train_pt(iteration)
                    self.train_ray(iteration)
                    self.global_epoch+=1

                    if (self.global_epoch+1)%self.args.epochs_per_valid == 0:
                        self.valid()                



    def valid_input(self, data_loader):
        _, (X,  Y) = data_loader.__next__()
        return X.to(self.device), Y.to(self.device)


    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable


    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl       



    def valid(self, model = "hnet", ray = None,  load = False, ckptname = "last", train_data = False):
        with torch.no_grad():
            if train_data:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[idxs]), batch_size=len(self.dict_user_train[idxs]), shuffle=False, drop_last=False))
                                 for idxs in range(self.args.num_users)]
            else:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]), batch_size=len(self.dict_user_test[idxs]), shuffle=False, drop_last=False))
                                 for idxs in range(self.args.num_users)]
            if load:
                self.load_hnet(ckptname)

            accs = {}
            loss_dict = {}
            loss_list = []
            aucs = {}
            for usr_id in self.users_used:
                X, Y = self.valid_input(data_loaders[usr_id])
                pred, loss = self.hnet(X, Y, usr_id, ray)
                acc, auc = self.acc_auc(pred, Y, is_training=False)
                accs[str(usr_id)] = acc
                loss_dict[str(usr_id)] = loss.item()
                loss_list.append(loss.item())
                aucs[str(usr_id)] = auc
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]
            kl_l_p = self.losses_r(loss_list, input_ray_numpy)
            self.pickle_record["valid"][str(self.global_epoch)] = {}
            self.pickle_record["valid"][str(self.global_epoch)]["losses"] = loss_dict
            self.pickle_record["valid"][str(self.global_epoch)]["accs"] = accs
            self.pickle_record["valid"][str(self.global_epoch)]["aucs"] = aucs
            self.pickle_record["valid"][str(self.global_epoch)]["a"] = kl_l_p
            self.pickle_record["valid"][str(self.global_epoch)]["input_ray"] = input_ray_numpy           
            self.logger.info("valid : iteration: {}, valid-loss : {}, valid-acc: {}, \
                losses: {}, input_ray: {}, a {},  accs:{}, aucs:{}.".format(self.global_epoch, \
                loss_dict[str(self.target_usr)], accs[str(self.target_usr)] , loss_dict, input_ray_numpy, kl_l_p, accs, aucs))



    def save_hnet(self, ckptname = None):
        states = {'epoch':self.global_epoch,
                  'model':self.hnet.state_dict(),
                  'optim':self.optim.state_dict(),
                  'input_ray': self.hnet.input_ray.data}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        os.makedirs(self.args.hnet_model_dir, exist_ok = True)
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> hnet saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))  


    def load_hnet(self, ckptname='last'):
        if ckptname == 'last':
            ckpts = os.listdir(self.args.hnet_model_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.global_epoch = checkpoint['epoch']
            self.hnet.load_state_dict(checkpoint['model'])
            self.hnet.input_ray.data = checkpoint["input_ray"].data.view(1, -1)
            self.optim.load_state_dict(checkpoint['optim'])
            self.logger.info("=> hnet loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))
