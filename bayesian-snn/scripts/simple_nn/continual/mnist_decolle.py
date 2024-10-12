import argparse
import torch
import sys
from itertools import islice, chain, tee
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import fnmatch
import time
import os
from data.utils import make_mnist_dataloader


def make_experiment_dir(path, exp_type):
    prelist = np.sort(fnmatch.filter(os.listdir(path), '[0-9][0-9][0-9]__*'))
    if len(prelist) == 0:
        expDirN = "001"
    else:
        expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

    results_path = time.strftime(path + r'/' + expDirN + "__" + "%d-%m-%Y",
                                 time.localtime()) + '_' + exp_type

    os.makedirs(results_path)

    return results_path


class NN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.thresh = nn.Parameter(torch.rand(1,hidden_size))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.thresh*output
        output = F.relu(output)
        output = self.out(output)
        return output

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default="results/rz0012", type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_period', type=int, default=1)
    parser.add_argument('--num_ite', default=3, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--thr', default=1.25, type=float)
    parser.add_argument('--scale_grad', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--T', type=int, default=1)

    parser.add_argument('--with_coresets', action='store_true', default=True)
    parser.add_argument('--coreset_length', type=int, default=15,
                        help='Number of batches in coreset')

    parser.add_argument('--num_samples_test', default=10, type=int)
    parser.add_argument('--rho', type=float, default=0.000075)
    parser.add_argument('--burn_in', type=int, default=10)

    parser.add_argument('--fixed_prec', action='store_true', default=True)
    parser.add_argument('--initial_prec', default=1., type=float)
    parser.add_argument('--prior_m', type=float, default=0.)
    parser.add_argument('--prior_s', default=1., type=float)

    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--tau', default=1., type=float)
    parser.add_argument('--prior', default=0.5, type=float)

    parser.add_argument('--device', type=int, default=None)

    args = parser.parse_args()



    results_path = make_experiment_dir(args.home + '/results',
                                       'mnist_bayesian_decolle_nepochs_%d_'
                                       % args.num_epochs)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # Create dataloaders
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    digits_all = [item for sublist in tasks for item in sublist]
    
    for ite in range(args.num_ite):
        tasks_seen = []

        acc_best = 0
        
        net = NN(28*28, 1024, 10).cuda()
        ce = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   


        for i, digits in enumerate(tasks):
            train_dl, test_dl_task \
                = make_mnist_dataloader(digits, args.batch_size, args.T)

            if len(tasks_seen) > 0:
                if args.with_coresets:
                    coresets = []
                    for task in tasks_seen:
                        train_dl_task_seen, _ \
                            = make_mnist_dataloader(task,
                                                    args.batch_size,
                                                    args.T)

                        train_iterator_seen = islice(iter(train_dl_task_seen),
                                                     args.coreset_length)
                        coreset_task = tee(train_iterator_seen, args.num_epochs)
                        coresets.append(coreset_task)


            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Digits: ' + str(digits))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

            for epoch in range(args.num_epochs):
                print('Epoch %d / %d' % (epoch + 1, args.num_epochs))

                if (len(tasks_seen) > 0) and args.with_coresets:
                    train_iterator = iter(train_dl)
                    for coreset in coresets:
                        train_iterator = chain(train_iterator, coreset[epoch])
                else:
                    train_iterator = iter(train_dl)
                net.train()
                loss = 0
                count = 0
                for (inputs, label) in tqdm(train_iterator):  # training loop
                    inputs = inputs.cuda()
                    label = label.cuda()
                    for t in range(inputs.shape[-1]):
                        optimizer.zero_grad()
                        cls = net(inputs[..., t])
                        l = ce(cls, label)
                        l.backward()
                        optimizer.step()
                        loss += l
                        count+=1
                        
                loss = loss/(count*args.batch_size)
                print(f'loss: {loss}')
                if (epoch + 1) % args.test_period == 0:
                    print('Prediction when network is forced to predict')
                    correct = 0
                    total = 0
                    for j, data in enumerate(test_dl_task):
                        images, labels = data
                        predicted = net(images[:,:,0].view(-1,28*28).cuda())
                        total += labels.size(0)
                        _, predicted = torch.max(predicted.data, 1)
                        correct += (predicted == labels.cuda()).sum().item()
                    print('Acc at epoch %d for current task: %f' % (epoch + 1, correct / total))
                    

                    if len(tasks_seen) > 0:
                        print('Testing on previously seen digits...')
                        for j, task in enumerate(tasks_seen):
                            _, test_dl_seen_task = make_mnist_dataloader(task, args.batch_size,args.T)
                            print('Prediction when network is forced to predict')
                            correct = 0
                            total = 0
                            for j, data in enumerate(test_dl_seen_task):
                                images, labels = data
                                predicted = net(images[:,:,0].view(-1,28*28).cuda())
                                total += labels.size(0)
                                _, predicted = torch.max(predicted.data, 1)
                                correct += (predicted == labels.cuda()).sum().item()
                            print('task: '+ f'{task}' +' acc at epoch %d: %f' % (
                                    epoch + 1, correct / total))
                            
                            
            tasks_seen += [digits]