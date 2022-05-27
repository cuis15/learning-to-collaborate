# learning-to-collaborate

## the implementation of learning to collaborate

---
### The description of all files
1. main.py: the main function for all experiments;
2. hyper_model/models.py: this script defines all necessary model structures;
3. solvers: this script implements two different optimization for learning the whole Pareto Front (linear and EPO);
4. train.py: this script defines the train function for all experiments (including evaluation, saving model, loading model, etc.);
5. utils/utils_data.py: this script pre-processes all data set which will be used for the following training and evaluating;
6. utils/utils_func.py: the needed extra functions;
7. utils/utils_sampling.py: this script is used for generating non i.i.d data for all experiments.
---

## Preparations

### Construct Conda Environment
python 3.6, the needed environment libraries are in requirements.txt
```
conda create -n E8519 python=3.6
conda install --yes --file requirements.txt
```

### Datasets
1. **Synthetic data set**
    the source code will generate the needed data when running the synthetic experiments
2. **Adult data set**
    Adult data set is pre-processed following the work in [1,2]
    the processed data set is in data/Adult
3. **CIFAR10**
    We download CIFAR10 when firstly running the CIFAR10 experiments and the downloaded data will be saved in data/CIFAR10
4. **eICU**
    eICU dataset needs approval when researchers need to have access to it.

## Get Started

### Parameters Description in main.py for Running all experiments
1. dataset: the needed dataset for running experiments;
2. trainN: the generated synthetic data samples for training;
3. std: the $\rho$ when generating synthetic data;
4. sample_ray: whether we need to sample direction vectors $d$ for training the Pareto Front of all objectives;
5. target_usr: the target client where the learned personalized model will deploy;
6. total_hnet_epoch: the num of epoch for training the Pareto Front;
7. total_ray_epoch: the num of epoch for training the direction vector $d$
8. lr: learning rate for updating the hypernetwork;
9. lr_prefer: learning rate for updating the direction vector $d$;
10. gpus: the GPU device;
11. n_hidden: the num of hidden layers of the hypernetwork;
12. num_users: the num of clients in each experiment

### Example Synthetic Experiment

```
python main.py --dataset synthetic --trainN 2000 --std 0.1 --sample_ray --target_dir synthetic --target_usr 4 --total_hnet_epoch 1000 --epochs_per_valid 1 --total_ray_epoch 200 --total_epoch 1 --gpus 0 --n_hidden 1 --lr 0.01 --lr_prefer 0.01 --seed 1 --solver_type linear
```

### Example Adult Experiment

```
python  main.py --dataset adult --target_dir adult --target_usr 0 --total_hnet_epoch 20 --epochs_per_valid 100 --total_ray_epoch 1 --total_epoch 2000 --gpus 0 --n_hidden 3 --lr 0.05 --lr_prefer 0.01 --seed 0 --solver_type epo
```

### Example CIFAR10 Experiment

```
python main.py --dataset cifar10 --num_users 10 --target_usr 9 --total_hnet_epoch 10000 --total_ray_epoch 1000 --total_epoch 1 --seed 3 --local_bs 512 --lr 0.01 --lr_prefer 0.01 --solver_type linear --sample_ray --n_hidden 3 --embedding_dim 5 --input_dim 20 --output_dim 2 --hidden_dim 100  --gpus 0
```


## Reference
[1] Tian Li, Maziar Sanjabi, Ahmad Beirami, and Virginia Smith. Fair resource allocation in federated learning.429arXiv preprint arXiv:1905.10497, 2019
[2] Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. InInternational431Conference on Machine Learning, pages 4615–4625. PMLR, 2019.
