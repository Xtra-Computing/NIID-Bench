import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from models import resnet_split_model 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--model_type', type=str, default='resnet18', help='neural network sub type')
    parser.add_argument('--cut_a', type=int, default=3, help='AdhocSL first cut layer')
    parser.add_argument('--cut_b', type=int, default=7, help='AdhocSL second cut layer')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='umber of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon/adhocSL')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--warmup', type=int, default=-1, help="warmup")
    parser.add_argument('--sl_step', type=int, default=5, help="frequency")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                
                # ------ NEW: ADHOC configuration -----
                elif args.alg == 'adhocSL':
                    if args.model == "resnet":
                        net = resnet_split_model.get_resnet_split(n_classes, args.cut_a, args.cut_b, args.model_type)
                    elif args.model == 'simple-cnn':
                        if args.dataset in ("mnist", 'femnist', 'fmnist'):
                            net = get_simpleCNNMINST_split(args.cut_a, args.cut_b, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                        else:
                            net = get_simpleCNN_split(args.cut_a, args.cut_b, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)

                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        # FOR REPRODUCABILITY
                        #net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                        net = get_simpleCNN_split(args.cut_a, args.cut_b, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        # FOR REPRODUCABILITY
                        #net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                        net = get_simpleCNNMINST_split(args.cut_a, args.cut_b, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    #net = ResNet50_cifar10()
                    net = resnet_split_model.get_resnet_split(n_classes, args.cut_a, args.cut_b, args.model_type)
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net
    model_meta_data = []
    layer_type = []
    
    ###### FOR REPRODUCABILITY
    for (k, v) in nets[0][0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

    for (k, v) in nets[0][1].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    for (k, v) in nets[0][2].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    '''
    if args.alg == 'adhocSL':
        for (k, v) in nets[0][0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        for (k, v) in nets[0][1].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        for (k, v) in nets[0][2].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

    else:
        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
    '''
    ###### FOR REPRODUCABILITY END
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu", adhoc=False, data_sharing=False, helpers=[]):
    logger.info('Training network %s' % str(net_id))
    if data_sharing:
        train_acc = compute_accuracy(net[net_id], train_dataloader, device=device, adhoc=adhoc)
        test_acc, conf_matrix = compute_accuracy(net[net_id], test_dataloader, get_confusion_matrix=True, device=device, adhoc=adhoc)

        tempModels, _, _ = init_nets(args.net_config, 0, 1, args)
        tempModel = tempModels[0]
    else:
        train_acc = compute_accuracy(net, train_dataloader, device=device, adhoc=adhoc)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, adhoc=adhoc)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        if adhoc:
            if data_sharing:
                logger.info('Data sharing round')
                optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[net_id][1].parameters()), lr=lr, weight_decay=args.reg)
                optimizer_a =[]
                optimizer_c = []
                '''
                for i in range(len(helpers)):
                    optimizer_a.append(optim.Adam(filter(lambda p: p.requires_grad, net[helpers[i]][0].parameters()), lr=lr, weight_decay=args.reg))
                    optimizer_c.append(optim.Adam(filter(lambda p: p.requires_grad, net[helpers[i]][2].parameters()), lr=lr, weight_decay=args.reg))
                '''
            else:
                optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg)
                optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg)
                optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg)
        else:
            # FOR REPRODUCABILITY
            #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
            optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg)
            optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg)
            optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        if adhoc:
            if data_sharing:
                logger.info('Data sharing round')
                optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[net_id][1].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
                optimizer_a =[]
                optimizer_c = []
                '''
                for i in range(len(helpers)):
                    optimizer_a.append(optim.Adam(filter(lambda p: p.requires_grad, net[helpers[i]][0].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True))
                    optimizer_c.append(optim.Adam(filter(lambda p: p.requires_grad, net[helpers[i]][2].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True))
                '''
            else:
                optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
                optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
                optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        else:
            # FOR REPRODUCABILITY
            #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
            #                   amsgrad=True)
            optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
            optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
            optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
    elif args_optimizer == 'sgd':
        if adhoc:
            if data_sharing:
                logger.info('Data sharing round')
                optimizer_b = optim.SGD(filter(lambda p: p.requires_grad, net[net_id][1].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
                optimizer_a =[]
                optimizer_c = []

                '''
                for i in range(len(helpers)):
                    optimizer_a.append(optim.SGD(filter(lambda p: p.requires_grad, net[helpers[i]][0].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg))
                    optimizer_c.append(optim.SGD(filter(lambda p: p.requires_grad, net[helpers[i]][2].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg))
                '''
            else:
                optimizer_a = optim.SGD(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
                optimizer_b = optim.SGD(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
                optimizer_c = optim.SGD(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        else:
            # FOR REPRODUCABILITY
            #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
            optimizer_a = optim.SGD(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
            optimizer_b = optim.SGD(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
            optimizer_c = optim.SGD(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()
    num_helpers = len(helpers)
    for epoch in range(epochs):
        epoch_loss_collector = []
        i_helper = 0
        if data_sharing:
            for tmps in zip(*train_dataloader):
                batch_size = max([tmps[i][0].size()[0] for i in range(num_helpers)])  # TODO: THIS NEEDS CHECK
                portion = int(batch_size/num_helpers)
                if portion == 0:
                    portion = 1
                iterations = int(batch_size/portion)
                # get the data samples
                x_s = []
                targets = []

                for i_helper in range(num_helpers):
                    x, target = tmps[i_helper]
                    x, target = x.to(device), target.to(device)
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    x_s.append(x)
                    targets.append(target)
                for it in range(iterations):
                    #print(f'It is {it}')
                    optimizer_b.zero_grad()
                    
                    start_a = it*portion
                    end_a = start_a + portion
                    # forward to helpers model part a
                    det_out_as = []
                    for i_helper in range(num_helpers):
                        net_params =  net[i_helper][0].state_dict()
                        tempModel[0].load_state_dict(net_params)
                        end_a_ = end_a
                        if len(targets[i_helper]) <  end_a:
                            end_a_ = len(x_s[i_helper])
                        out_a = tempModel[0](x_s[i_helper][start_a:end_a_])
                        det_out_a = out_a.clone().detach().requires_grad_(True)
                        det_out_as.append(det_out_a)
                    
                    # concate activations and forward to model part b
                    det_out_a_all = torch.cat(det_out_as)
                    out_b = net[net_id][1](det_out_a_all)
                    det_out_b = out_b.clone().detach().requires_grad_(True)

                    # forward to helpers model part c
                    grad_bs = []
                    loss_ = 0

                    start = 0
                    portion_ = 0
                    for i_helper in range(num_helpers):
                        net_params =  net[i_helper][2].state_dict()
                        tempModel[2].load_state_dict(net_params)
                        start = start + portion_#i_helper*portion
                        end = start + portion
                        if len(targets[i_helper]) <  end_a:
                            if  len(targets[i_helper]) - start_a > 0:
                                end = start + len(targets[i_helper]) - start_a
                            else:
                                end = start
                        portion_ = end - start
                        
                        det_out_b_ = det_out_b[start:end].clone().detach().requires_grad_(True)
                        out = tempModel[2](det_out_b_)
                        
                        end_a_ = end_a
                        if len(targets[i_helper]) <  end_a:
                            end_a_ = len(targets[i_helper])
                        if targets[i_helper][start_a:end_a_].size()[0] == 0 or out.size()[0] == 0:
                            continue
                        loss = criterion(out, targets[i_helper][start_a:end_a_])
                        loss.backward()
                        
                        loss_ += loss.item()                    
                        grad_b = det_out_b_.grad.clone().detach()
                        grad_bs.append(grad_b)
                    
                    # concate the gradients and backprop to model part b
                    grad_b_all = torch.cat(grad_bs)
                    out_b.backward(grad_b_all)
                    optimizer_b.step()
                     
                    cnt += 1
                    loss__ = loss_/num_helpers
                    epoch_loss_collector.append(loss__)
            
            # end of epoch
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
            train_acc = compute_accuracy(net[net_id], train_dataloader, device=device, adhoc=adhoc)
            test_acc, conf_matrix = compute_accuracy(net[net_id], test_dataloader, get_confusion_matrix=True, device=device, adhoc=adhoc)
        else:
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    if adhoc:
                        optimizer_b.zero_grad()
                        optimizer_a.zero_grad()
                        optimizer_c.zero_grad()
                    else:
                        # FOR REPRODUCABILITY
                        #optimizer.zero_grad()

                        optimizer_b.zero_grad()
                        optimizer_a.zero_grad()
                        optimizer_c.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    if adhoc:
                            out_a = net[0](x)
                            det_out_a = out_a.clone().detach().requires_grad_(True)

                            out_b = net[1](det_out_a)
                            det_out_b = out_b.clone().detach().requires_grad_(True)

                            out = net[2](det_out_b)
                    else:
                        # FOR REPRODUCABILITY
                        #out = net(x)
                        out_a = net[0](x)
                        det_out_a = out_a.clone().detach().requires_grad_(True)

                        out_b = net[1](det_out_a)
                        det_out_b = out_b.clone().detach().requires_grad_(True)

                        out = net[2](det_out_b)
                        
                    loss = criterion(out, target)

                    loss.backward()

                    if adhoc:
                        optimizer_c.step()

                        grad_b = det_out_b.grad.clone().detach()
                        out_b.backward(grad_b)
                        optimizer_b.step()

                        grad_a = det_out_a.grad.clone().detach()
                        out_a.backward(grad_a)

                        optimizer_a.step()
                    else:
                        # FOR REPRODUCABILITY
                        #optimizer.step()
                        optimizer_c.step()

                        grad_b = det_out_b.grad.clone().detach()
                        out_b.backward(grad_b)
                        optimizer_b.step()

                        grad_a = det_out_a.grad.clone().detach()
                        out_a.backward(grad_a)

                        optimizer_a.step()

                    cnt += 1
                    epoch_loss_collector.append(loss.item())
            
            # end of epoch
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
            train_acc = compute_accuracy(net, train_dataloader, device=device, adhoc=adhoc)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, adhoc=adhoc)  

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    
    if adhoc:
        if data_sharing: #TODO: fix this...
            
            pass
        else:
            net[0].to(device)
            net[1].to(device)
            net[2].to(device)
    else:
        #net.to(device)
        net[0].to(device)
        net[1].to(device)
        net[2].to(device)

    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        # FOR REPRODUCABILITY
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg)
        optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg)
        optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg)

    elif args_optimizer == 'amsgrad':
        # FOR REPRODUCABILITY
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
        #                   amsgrad=True)
        optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, weight_decay=args.reg,
                            amsgrad=True)
        optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, weight_decay=args.reg,
                            amsgrad=True)
        optimizer_c = optim.Adam(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, weight_decay=args.reg,
                            amsgrad=True)
    elif args_optimizer == 'sgd':
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        optimizer_a = optim.SGD(filter(lambda p: p.requires_grad, net[0].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        optimizer_b = optim.SGD(filter(lambda p: p.requires_grad, net[1].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        optimizer_c = optim.SGD(filter(lambda p: p.requires_grad, net[2].parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    # FOR REPRODUCABILITY
    #global_weight_collector = list(global_net.to(device).parameters())
    global_weight_collector = list(global_net[0].to(device).parameters())
    global_weight_collector.append(list(global_net[1].to(device).parameters()))
    global_weight_collector.append(list(global_net[2].to(device).parameters()))

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            # FOR REPRODUCABILITY
            #optimizer.zero_grad()

            optimizer_b.zero_grad()
            optimizer_a.zero_grad()
            optimizer_c.zero_grad()

            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            # FOR REPRODUCABILITY
            #out = net(x)
            out_a = net[0](x)
            det_out_a = out_a.clone().detach().requires_grad_(True)

            out_b = net[1](det_out_a)
            det_out_b = out_b.clone().detach().requires_grad_(True)

            out = net[2](det_out_b)

            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            param_all = 0
            for param_index, param in enumerate(net[0].parameters()):
                param_all += 1
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            for param_index, param in enumerate(net[1].parameters()):
                param_all += 1
                fed_prox_reg += ((mu / 2) * torch.norm((param_all - global_weight_collector[param_index]))**2)
            for param_index, param in enumerate(net[2].parameters()):
                param_all += 1
                fed_prox_reg += ((mu / 2) * torch.norm((param_all - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            # FOR REPRODUCABILITY
            #optimizer.step()
            optimizer_c.step()

            grad_b = det_out_b.grad.clone().detach()
            out_b.backward(grad_b)
            optimizer_b.step()

            grad_a = det_out_a.grad.clone().detach()
            out_a.backward(grad_a)

            optimizer_a.step()


            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)
    
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    #net.to(device)
    net[0].to(device)
    net[1].to(device)
    net[2].to(device)
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para

def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):

    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu", data_sharing=False, helpers=[]):
    avg_acc = 0.0

    step = 0
    
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        
        if args.alg == 'adhocSL':
            net[0].to(device)
            net[1].to(device)
            net[2].to(device)
            adhoc = True
        else:
            #net.to(device)
            #FOR REPRODUCABILITY
            net[0].to(device)
            net[1].to(device)
            net[2].to(device)
            adhoc = False

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            if data_sharing:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local = []
                for i in range(len(helpers[net_id])):
                    dataidxs_ = net_dataidx_map[helpers[net_id][i]]
                    train_dl_local_, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_, noise_level)
                    train_dl_local.append(train_dl_local_)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if data_sharing:
            trainacc, testacc = train_net(net_id, nets, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device, adhoc=adhoc, data_sharing=True, helpers=helpers[net_id])
        else:
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device, adhoc=adhoc)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        step += 1
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        #net.to(device)
        #FOR REPRODUCABILITY
        net[0].to(device)
        net[1].to(device)
        net[2].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models=[]
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list



def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

#----- NEW AD-HOC configuration -----
def find_helpers(dataset, net_dataidx_map, n_parties, traindata_cls_counts):
    K = 0
    if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        K = 2
    else:
        K = 10
    if dataset == "cifar100":
        K = 100
    elif dataset == "tinyimagenet":
        K = 200

    helpers = {}
    for i in range(n_parties):
        helpers.update({i:[i]})
    for k in range(K):
        print(f'attribute {k}')
        times = [0 for i in range(n_parties)]
        for i in range(n_parties):
            distribution = traindata_cls_counts[i]
            if k in distribution.keys():
                times[i] = distribution[k]
        print(times)
        order_ = np.argsort(times) # increasing order
        order = order_[::-1 ] # non-increasing order
        sorted_times = sorted(times, reverse=True)
        print(sorted_times)
        itr = -1
        for j in range(n_parties):
            if sorted_times[j] == 0:
                itr  = j
                break
        threshold = -1
        # option-1 policy for threshold
        if sorted_times[itr] == 0:
            threshold = itr
        else:
            threshold = int(n_parties/2) # to do find something more smart
        '''
        if itr < int(n_parties/2):
            threshold = itr
        else:
            threshold = int(n_parties/2)
        '''
        order_helpers = 0
        for j in range(threshold, n_parties):
            need_help = order[j]
            send_help = order[order_helpers]
            order_helpers = order_helpers + 1
            if order_helpers >= threshold:
                order_helpers = 0
            
            if send_help not in helpers[need_help]:
                helpers[need_help].append(send_help)
    print(helpers)
    return helpers

# MAIN
if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)
    print(f'HERE IS {args.alg}')
    #------- New code for ADHOC configuration -----
    if args.alg == 'adhocSL':
        print("Running adhocSL algorithm.")
        
        warmup = args.warmup
        sl_step = args.sl_step
    

        # initialize the communication graph for the sl-rounds
        graph_comm = find_helpers(args.dataset, net_dataidx_map, args.n_parties, traindata_cls_counts)

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para_a = global_model[0].state_dict()
        global_para_b = global_model[1].state_dict()
        global_para_c = global_model[2].state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net[0].load_state_dict(global_para_a)
                net[1].load_state_dict(global_para_b)
                net[2].load_state_dict(global_para_c)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para_a = global_model[0].state_dict()
            global_para_b = global_model[1].state_dict()
            global_para_c = global_model[2].state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx][0].load_state_dict(global_para_a)
                        nets[idx][1].load_state_dict(global_para_b)
                        nets[idx][2].load_state_dict(global_para_c)
            else:
                for idx in selected:
                    nets[idx][0].load_state_dict(global_para_a)
                    nets[idx][1].load_state_dict(global_para_b)
                    nets[idx][2].load_state_dict(global_para_c)

            if warmup == -1:
                data_sharing = False
            elif ((round >= warmup) and (round % sl_step ==0)):
                data_sharing = True
            else:
                data_sharing = False
            
            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device, data_sharing=data_sharing, helpers=graph_comm)

            # update global model
            # Question: In case of data sharing we take these into account??
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            # Aggregation
            for idx in range(len(selected)):
                net_para_a = nets[selected[idx]][0].cpu().state_dict()
                net_para_b = nets[selected[idx]][1].cpu().state_dict()
                net_para_c = nets[selected[idx]][2].cpu().state_dict()
                if idx == 0:
                    for key in net_para_a:
                        global_para_a[key] = net_para_a[key] * fed_avg_freqs[idx]
                    for key in net_para_b:
                        global_para_b[key] = net_para_b[key] * fed_avg_freqs[idx]
                    for key in net_para_c:
                        global_para_c[key] = net_para_c[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para_a:
                        global_para_a[key] += net_para_a[key] * fed_avg_freqs[idx]
                    for key in net_para_b:
                        global_para_b[key] += net_para_b[key] * fed_avg_freqs[idx]
                    for key in net_para_c:
                        global_para_c[key] += net_para_c[key] * fed_avg_freqs[idx]
            
            global_model[0].load_state_dict(global_para_a)
            global_model[1].load_state_dict(global_para_b)
            global_model[2].load_state_dict(global_para_c)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            
            global_model[0].to(device)
            global_model[1].to(device)
            global_model[2].to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device, adhoc=True)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, adhoc=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        print("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        print('cool')
        '''
        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        '''
        global_para_a = global_model[0].state_dict()
        global_para_b = global_model[1].state_dict()
        global_para_c = global_model[2].state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net[0].load_state_dict(global_para_a)
                net[1].load_state_dict(global_para_b)
                net[2].load_state_dict(global_para_c)
        print("start round")
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            #global_para = global_model.state_dict()
            global_para_a = global_model[0].state_dict()
            global_para_b = global_model[1].state_dict()
            global_para_c = global_model[2].state_dict()

            '''
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
            '''
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx][0].load_state_dict(global_para_a)
                        nets[idx][1].load_state_dict(global_para_b)
                        nets[idx][2].load_state_dict(global_para_c)
            else:
                for idx in selected:
                    nets[idx][0].load_state_dict(global_para_a)
                    nets[idx][1].load_state_dict(global_para_b)
                    nets[idx][2].load_state_dict(global_para_c)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model[0].to('cpu')
            global_model[1].to('cpu')
            global_model[2].to('cpu')
            
            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
            # Aggregation
            for idx in range(len(selected)):
                net_para_a = nets[selected[idx]][0].cpu().state_dict()
                net_para_b = nets[selected[idx]][1].cpu().state_dict()
                net_para_c = nets[selected[idx]][2].cpu().state_dict()
                if idx == 0:
                    for key in net_para_a:
                        global_para_a[key] = net_para_a[key] * fed_avg_freqs[idx]
                    for key in net_para_b:
                        global_para_b[key] = net_para_b[key] * fed_avg_freqs[idx]
                    for key in net_para_c:
                        global_para_c[key] = net_para_c[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para_a:
                        global_para_a[key] += net_para_a[key] * fed_avg_freqs[idx]
                    for key in net_para_b:
                        global_para_b[key] += net_para_b[key] * fed_avg_freqs[idx]
                    for key in net_para_c:
                        global_para_c[key] += net_para_c[key] * fed_avg_freqs[idx]
            
            '''
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            global_model.to(device)
            '''
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, moon_model=True, device=device)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)
# END MAIN