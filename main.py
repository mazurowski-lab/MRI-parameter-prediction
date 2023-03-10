# predict image acquisition parameters (IAPs) directly from breast MR images
import os
if os.getcwd() != 'workspace' and os.path.exists('workspace'):
    os.chdir('workspace')

from src.dataset import *
from src.utils import *
from src.vizutils import *
from src.IAP_model import get_total_criterion

import os
import random
from tqdm import tqdm
import datetime

# torch
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

# GPUs
device_ids = [1] # indices of devices for models, data and otherwise
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('running on {}'.format(device))

# set random seed
seed = 1337
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# config:
# data and model choice
dataset_name = 'dbc_by_scanner'
labeling = 'all'

train_size = 10000
val_size = 2000
test_size = 2000

model = resnet18

checkpoint_paths = {
}

# training options
train = True
batch_size_factors = {
          'resnet18' : 512
}

img_size = 224
train_batchsize = batch_size_factors[model.__name__] * len(device_ids)
eval_batchsize = 64

checkpoint_path_prev = None
train_with_augmentations = False
save_checkpoints = True
checkpoint_setting = 'incremental'

epochs = 100
checkpoint_dir = "saved_models/feature_pred_all/{}".format(dataset_name)

optimizer = torch.optim.Adam(net.parameters(),
                            lr= 0.001,# default=0.001
                            weight_decay=0.0001     
                        )

start_epoch = 0

# option for other experiments
test_this_IAP_only = None

def main():
    # load dataset and loader (note: this will take a minute or so to run)

    if train_with_augmentations:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=img_size)
        ])
        print('training with augmentations')
    else:
        train_transform = transforms.Compose([])

    trainset, valset, testset = get_datasets(dataset_name, 
                                    train_size=train_size, 
                                    val_size=val_size,
                                    test_size=test_size,
                                    labeling=labeling,
                                    return_filenames=True,
                                    different_cases_for_train_val_test=True,
                                    regress_dense_IAPs=True
                                )


    trainloader = DataLoader(trainset, 
                            batch_size=train_batchsize, 
                            shuffle=True)
    valloader = DataLoader(valset, 
                        batch_size=eval_batchsize)
    testloader = DataLoader(testset, 
                        batch_size=eval_batchsize)

    # load model
    net = model()
    # fix first lyr
    make_netinput_onechannel(net, model)

    try:
        all_feature_names = trainset.dataset.dataset.all_feature_names
    except AttributeError:
        all_feature_names = trainset.dataset.all_feature_names
    num_output_features = np.array(list(all_feature_names.values())).sum()
    print(num_output_features)
    change_num_output_features(net, model, num_output_features)

    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids = range(len(device_ids)))

    # set up custom loss
    total_criterion = get_total_criterion(all_feature_names, device)

    if train:
        # training
        epochs = 100
        checkpoint_dir = "saved_models/feature_pred_all/{}".format(dataset_name)

        optimizer = torch.optim.Adam(net.parameters(),
                                    lr= 0.001,# default=0.001
                                    weight_decay=0.0001     
                                )

        start_epoch = 0
        best_val_loss = np.inf
        for epoch in range(start_epoch, epochs):
            net.train()
            print("Epoch {}:".format(epoch))

            total_examples = 0

            train_loss = 0
            train_feature_losses = {feature_name : 0 for feature_name in all_feature_names.keys()}

            # train for one epoch
            for batch_idx, (inputs, targets, _) in tqdm(enumerate(trainloader), total=len(trainloader.dataset)//train_batchsize):
                inputs = inputs.to(device)

                # apply transformations
                inputs = train_transform(inputs)

                # reset gradients
                optimizer.zero_grad()

                # inference
                outputs = net(inputs)

                # backprop
                total_loss, feature_losses, _ = total_criterion(outputs, targets)
                total_loss.backward()

                # iterate
                optimizer.step()

                train_loss += total_loss
                train_feature_losses = {feature_name : train_feature_losses[feature_name] + feature_losses[feature_name] for feature_name, _ in feature_losses.items()}

            # results
            avg_loss = train_loss / (batch_idx + 1)
            print("Training loss: %.4f" %(avg_loss))
            # print("training loss per feature: ", {feature_name : train_feature_losses[feature_name] / (batch_idx + 1) for feature_name, feature_loss in train_feature_losses.items()})

            print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            print("Validation...")
            total_examples = 0

            net.eval()

            val_loss = 0
            val_feature_losses = {feature_name : 0 for feature_name in all_feature_names.keys()}

            # for accuracy computation
            total_examples = 0
            total_feature_correct_classification_examples = None 
            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(valloader):
                    # Copy inputs to device
                    inputs = inputs.to(device)
                    # Generate output from the DNN.
                    outputs = net(inputs)
                    
                    total_loss, feature_losses, feature_correct_classification_examples = total_criterion(outputs, targets)

                    # print(feature_correct_classification_examples)

                    val_loss += total_loss
                    val_feature_losses = {feature_name : val_feature_losses[feature_name] + feature_losses[feature_name] for feature_name, feature_loss in feature_losses.items()}

                    # for accuracy computation at the end
                    total_examples += eval_batchsize 

                    if not total_feature_correct_classification_examples:
                        # initialize correct example counts only for classification features
                        total_feature_correct_classification_examples = {feature_name : 0 for feature_name in feature_correct_classification_examples.keys()}
                    # add counts to totals
                    total_feature_correct_classification_examples = {feature_name : total_feature_correct_classification_examples[feature_name] + feature_correct_classification_examples[feature_name] for feature_name, _ in feature_correct_classification_examples.items()}

                    # print('running total correct example counts:', total_feature_correct_classification_examples, '\n')

            avg_loss = val_loss / len(valloader)
            print("validation loss: %.4f" %(avg_loss))

            # show metrics per feature (convert CE losses to accΩuracies)
            # avg all losses over batch
            val_metrics_per_feature = {feature_name : val_feature_losses[feature_name] / (batch_idx + 1) for feature_name, feature_loss in val_feature_losses.items()}

            # convert CE losses to accuracies
            for feature_name in feature_correct_classification_examples.keys():
                avg_acc = total_feature_correct_classification_examples[feature_name] / total_examples
                val_metrics_per_feature[feature_name] = avg_acc

            print("validation metrics per feature: ", val_metrics_per_feature, '\n')

            # Save for checkpoint
            if save_checkpoints:
                if (avg_loss < best_val_loss and checkpoint_setting == 'best') or ((epoch % 5 == 0) and checkpoint_setting == 'incremental'):
                    best_val_loss = avg_loss
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    print("Saving ...")
                    state = {'net': net.state_dict(),
                                'epoch': epoch}

                    # delete older checkpoint
                    if checkpoint_setting == 'best' and checkpoint_path_prev:
                        os.remove(checkpoint_path_prev)

                    # save new checkpoint
                    best_val_metric = avg_loss
                    checkpoint_path = "{}_{}_{}_{}_{}_{}_best.h5".format(model.__name__,
                        len(trainset), len(valset), best_val_metric, epoch, labeling)
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
                    torch.save(state, checkpoint_path)

                    checkpoint_path_prev = checkpoint_path


    # test
    checkpoint_path = 'saved_models/feature_pred_all/dbc_by_scanner/...' # fill in here
    net.load_state_dict(torch.load(checkpoint_path)['net'])

    print("Testing...")
    total_examples = 0

    net.eval()

    test_loss = 0
    test_feature_losses = {feature_name : 0 for feature_name in all_feature_names.keys()}

    # regression results
    TE_gt = []
    TE_pred = []
    TR_gt = []
    TR_pred = []

    # for accuracy computation
    total_examples = 0
    total_feature_correct_classification_examples = None 
    topk_total_feature_correct_classification_examples = None 
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            # Generate output from the DNN.
            outputs = net(inputs)

            # print(outputs)

            # store regression results individually
            TE_pred += outputs[:, -2: -1].flatten().tolist()
            TR_pred += outputs[:, -1:].flatten().tolist()
            TE_gt += targets['TE (Echo Time)'].float().tolist()
            TR_gt += targets['TR (Repetition Time)'].float().tolist()
            
            total_loss, feature_losses, feature_correct_classification_examples, topk_feature_correct_classification_counts = total_criterion(outputs, targets, get_topk_counts=True)

            # print(feature_correct_classification_examples)

            test_loss += total_loss
            test_feature_losses = {feature_name : test_feature_losses[feature_name] + feature_losses[feature_name] for feature_name, feature_loss in feature_losses.items()}

            # for accuracy computation at the end
            total_examples += eval_batchsize 

            if not total_feature_correct_classification_examples:
                # initialize correct example counts only for classification features
                total_feature_correct_classification_examples = {feature_name : 0 for feature_name in feature_correct_classification_examples.keys()}
            # add counts to totals
            total_feature_correct_classification_examples = {feature_name : total_feature_correct_classification_examples[feature_name] + feature_correct_classification_examples[feature_name] for feature_name, _ in feature_correct_classification_examples.items()}

            # print('running total correct example counts:', total_feature_correct_classification_examples, '\n')

            # top k stuff
            if not topk_total_feature_correct_classification_examples:
                # initialize correct example counts only for classification features
                topk_total_feature_correct_classification_examples = {}
                for k in topk_feature_correct_classification_counts.keys():
                    topk_total_feature_correct_classification_examples[k] = {feature_name : 0 for feature_name in topk_feature_correct_classification_counts[k].keys()}
            
            # add counts to totals
            for k in topk_feature_correct_classification_counts.keys():
                topk_total_feature_correct_classification_examples[k] = {feature_name : topk_total_feature_correct_classification_examples[k][feature_name] + topk_feature_correct_classification_counts[k][feature_name] for feature_name, _ in topk_feature_correct_classification_counts[k].items()} 

    avg_loss = test_loss / len(testloader)
    print("test loss: %.4f" %(avg_loss))

    # show metrics per feature (convert CE losses to accuracies)
    # avg all losses over batch
    test_metrics_per_feature = {feature_name : test_feature_losses[feature_name] / (batch_idx + 1) for feature_name, feature_loss in test_feature_losses.items()}

    # convert CE losses to accuracies
    for feature_name in feature_correct_classification_examples.keys():
        avg_acc = total_feature_correct_classification_examples[feature_name] / total_examples
        test_metrics_per_feature[feature_name] = avg_acc

    print("test metrics per feature: ", test_metrics_per_feature, '\n')

    print("top k accuracies:")
    for k, topk_correct_classification_counts in topk_feature_correct_classification_counts.items():
        print("k = {}:".format(k))
        print(topk_correct_classification_counts)
        for feature_name in topk_correct_classification_counts.keys():
            avg_acc = topk_total_feature_correct_classification_examples[k][feature_name] / total_examples
            print(feature_name, avg_acc)