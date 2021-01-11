# Fed-Learning-non-iid-benchmark

This code runs a benchmark for federated learning algorithms under non-IID data distribution scenarios. Specifically, we implement 4 federated learning algorithms (FedAvg, FedProx, SCAFFOLD & FedNova), 3 types of non-IID settings (label distribution skew, feature distribution skew & quantity skew) and 6 datasets (MNIST, Cifar-10, Fashion-MNIST, SVHN, Generated 3D dataset, FEMNIST).

## Non-IID Settings
### Label Distribution Skew
* **Quantity-based label imbalance**: each party owns data samples of a fixed number of labels.
* **Distribution-based label imbalance**: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
### Feature Distribution Skew
* **Noise-based feature imbalance**: 

Here is one example to run this code:
```
python experiments.py --model=simple-cnn \
    --dataset=femnist \
    --alg=scaffold \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=80 \
    --n_parties=10 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=real \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0
```

