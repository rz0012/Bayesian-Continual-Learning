import argparse
import torch
import sys
from itertools import islice, chain, tee
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
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
        # self.thresh = nn.Parameter(torch.rand(1,hidden_size))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        # output = self.thresh*output
        output = F.relu(output)
        output = self.out(output)
        return output

def model(x_data, y_data):
    
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module().cuda()
    
    lhat = log_softmax(lifted_reg_model(x_data)).cuda()
    
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data).cuda()

def guide(x_data, y_data):
    softplus = torch.nn.Softplus().cuda()
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module().cuda()


num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.argmax(mean, dim=1)

num_samples = 10
def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().cpu().numpy() for model in sampled_models]
    return np.asarray(yhats)


def test_batch(images, labels):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0
    
    for i in range(len(labels)):
        all_digits_prob = []
        highted_something = False
        for j in range(len(classes)):
        
            highlight=False
        
            histo = []
            histo_exp = []
        
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))
            
            prob = np.percentile(histo_exp, 50) #sampling median probability
        
            if(prob>0.2): #select if network thinks this sample is 20% chance of this being a label
                highlight = True #possibly an answer
        
            all_digits_prob.append(prob)
        
            if(highlight):
                highted_something = True

        predicted = np.argmax(all_digits_prob)
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                correct_predictions +=1.0

    return len(labels), correct_predictions, predicted_for_images 


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
        optim = Adam({"lr": 0.001})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())


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
                        loss += svi.step(inputs[..., t], label)
                        count+=1
                loss = loss/(count*args.batch_size)
                print(f'loss: {loss}')
                if (epoch + 1) % args.test_period == 0:
                    print('Prediction when network is forced to predict')
                    correct = 0
                    total = 0
                    for j, data in enumerate(test_dl_task):
                        images, labels = data
                        predicted = predict(images[:,:,0].view(-1,28*28).cuda())
                        total += labels.size(0)
                        correct += (predicted == labels.cuda()).sum().item()
                    print('Acc at epoch %d for current task: %f' % (epoch + 1, correct / total))
                    
                    print('Prediction when network can refuse')
                    correct = 0
                    total = 0
                    total_predicted_for = 0
                    for j, data in enumerate(test_dl_task):
                        images, labels = data
                        images = images[:,:,0].view(-1,28*28).cuda()
                        labels = labels.cuda()
                        total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels)
                        total += total_minibatch
                        correct += correct_minibatch
                        total_predicted_for += predictions_minibatch

                    print("Total images: ", total)
                    print("Skipped: ", total-total_predicted_for)
                    print("Accuracy when made predictions: " + str(correct / total_predicted_for))

                    if len(tasks_seen) > 0:
                        print('Testing on previously seen digits...')
                        for j, task in enumerate(tasks_seen):
                            _, test_dl_seen_task = make_mnist_dataloader(task, args.batch_size,args.T)
                            print('Prediction when network is forced to predict')
                            correct = 0
                            total = 0
                            for j, data in enumerate(test_dl_seen_task):
                                images, labels = data
                                predicted = predict(images[:,:,0].view(-1,28*28).cuda())
                                total += labels.size(0)
                                correct += (predicted == labels.cuda()).sum().item()
                            print('task: '+ f'{task}' +' acc at epoch %d: %f' % (
                                    epoch + 1, correct / total))
                            
                            
                            print('Prediction when network can refuse')
                            correct = 0
                            total = 0
                            total_predicted_for = 0
                            for j, data in enumerate(test_dl_seen_task):
                                images, labels = data
                                images = images[:,:,0].view(-1,28*28).cuda()
                                labels = labels.cuda()
                                total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels)
                                total += total_minibatch
                                correct += correct_minibatch
                                total_predicted_for += predictions_minibatch

                            print("Total images: ", total)
                            print("Skipped: ", total-total_predicted_for)
                            print("Accuracy when made predictions: " + str(correct / total_predicted_for))
                            
            tasks_seen += [digits]