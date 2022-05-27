# code for data prepare
from torchvision import datasets, transforms
from utils.utils_sampling import iid, noniid
import os
import pdb 
import json
import numpy as np 
import torch
import random

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])



def simple_data(args):

    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dict_users_test = {}

    if args.dataset == "adult":
        train_data_dir = os.path.join(args.data_root, "adult/train/mytrain.json")
        test_data_dir = os.path.join(args.data_root, "adult/test/mytest.json")

        with open(train_data_dir, "r") as f:
            train_data = json.load(f)["user_data"]
        dataset_train = [(np.array(x).astype(np.float32),float(y)) for x, y in zip(train_data["phd"]["x"], train_data["phd"]["y"])] + \
                        [(np.array(x).astype(np.float32),float(y)) for x, y in zip(train_data["non-phd"]["x"], train_data["non-phd"]["y"])]
        dict_users_train[0] = [i for i in range(len(train_data["phd"]["y"]))] 
        dict_users_train[1] = [i+len(dict_users_train[0]) for i in range(len(train_data["non-phd"]["y"])) ]

        with open(test_data_dir, "r") as f:
            test_data = json.load(f)["user_data"]
        dataset_test = [(np.array(x).astype(np.float32),float(y)) for x, y in zip(test_data["phd"]["x"], test_data["phd"]["y"])] + \
                        [(np.array(x).astype(np.float32),float(y)) for x, y in zip(test_data["non-phd"]["x"], test_data["non-phd"]["y"])]
        dict_users_test[0] = [i for i in range(len(test_data["phd"]["y"]))] 
        dict_users_test[1] = [i+len(dict_users_test[0]) for i in range(len(test_data["non-phd"]["y"])) ]

        return dataset_train, dataset_test, dict_users_train, dict_users_test       


    elif args.dataset == "synthetic": # generate dataset  dataset_train = [[x, y], [x, y],...,[x, y]]
                                      # dict_user_train is a dict {5:[1,2,3,...,10]}
        args.num_users = 6
        args.testN = int(0.5*args.trainN)
        v = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std**2 * np.eye(args.input_dim)

        for usr in [0,1,2,3,4,5]:
            tmp_trainN =  args.trainN
            tmp_testN =  args.testN 
            # if usr == 2:
            #     tmp_trainN = 10 * args.trainN
            #     tmp_testN = 10 * args.testN 
            # else:
            #     tmp_trainN = args.trainN
            #     tmp_testN = args.testN 
            r_0 = np.random.multivariate_normal(mean, cov)
            u_m = v + r_0

            # if usr !=3:
 
            x_m = np.random.uniform(-1.0, 1.0, (tmp_trainN+tmp_testN, args.input_dim))
            if usr in [0,1,2]:
                y_m = np.dot(x_m, u_m) + np.random.normal(0, args.sigma**2, (tmp_trainN+tmp_testN,))
            elif usr in [3,4,5]: 
                y_m = -np.dot(x_m, u_m) + np.random.normal(0, args.sigma**2, (tmp_trainN+tmp_testN,))
            else:
                print("error usr in generating synthetic data.")

            dataset_train.extend([ (x.astype(np.float32),y) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN]) ])
            dataset_test.extend([ (x.astype(np.float32),y) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:]) ])

            try:
                dict_users_train[usr] = [i+ dict_users_train[usr-1][-1]+1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ dict_users_test[usr-1][-1]+1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i+ 0  for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ 0  for i in range(tmp_testN)]              

        return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_root, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.data_root, train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=True, download=False, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=False, download=False, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)


    elif args.dataset == "adult" or args.dataset == "synthetic":
        
        dataset_train, dataset_test, dict_users_train, dict_users_test = simple_data(args)

    elif args.dataset == "eicu":
        dataset_train, dataset_test, dict_users_train, dict_users_test = eicu_data(args)

    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users_train, dict_users_test


