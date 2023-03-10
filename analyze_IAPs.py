# additional analysis: Analyze IAP distributions + statistics in dataset

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

# set up everything

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
          'resnet18' : 64
}
checkpoint_path_prev = None
train_with_augmentations = False
save_checkpoints = True
checkpoint_setting = 'incremental'
#checkpoint_setting = 'best'


test_this_IAP_only = None
# load dataset and loader
img_size = 224
train_batchsize = batch_size_factors[model.__name__] * len(device_ids)
eval_batchsize = 64

def main():
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
                                )


    trainloader = DataLoader(trainset, 
                            batch_size=train_batchsize, 
                            shuffle=True)
    valloader = DataLoader(valset, 
                        batch_size=eval_batchsize)
    testloader = DataLoader(testset, 
                        batch_size=eval_batchsize)
    
    ## (1) look at correlations between IAPs
    # create df of each dataset, where each row is a datapoint with certain IAP values
    data_dfs = {}
    for loader_idx, loader in enumerate([trainloader, valloader, testloader]):
        data_df = None
        for _, (inputs, targets, _) in enumerate(loader):
            new_data_df = pd.DataFrame.from_dict(targets)
            
            if data_df is None:
                data_df = new_data_df
            else:
                data_df = pd.concat([data_df, new_data_df])
                
        data_dfs[['train', 'val', 'test'][loader_idx]] = data_df  

    save_dir = 'IAP_analyses/correlations'
    for dset_name, data_df in data_dfs.items():
        print(dset_name)
        df_corr = data_df.corr(method='spearman')
        
        df_corr.to_csv(os.path.join(save_dir, '{}.csv'.format(dset_name)))

    
    ## (2) look at distributions of IAP values
    import matplotlib.pyplot as plt
    save_dir = 'IAP_analyses/IAP_dists'
    for dset_name, data_df in data_dfs.items():
        print(dset_name)
        plt.figure()
        df_corr = data_df.rename(columns={"Flip Angle \n": "Flip Angle"}).hist(figsize=(10,10), bins=25)
        plt.savefig((os.path.join(save_dir, '{}.pdf'.format(dset_name))))
        plt.show()


    ## (3) see if combinations of IAPs overlap between train and test sets
    """
    see the overlap between two subsets in terms of common IAP combinations (e.g. bw train and test)

    for subsets A and B: (1) how many unique combos only in A, (2) how many unique combos only in B, (3) how many unique combos in both
    """
    for dset_nameA, data_dfA in data_dfs.items():
        for dset_nameB, data_dfB in data_dfs.items():
            if dset_nameA != dset_nameB:
                # keep only unique IAP combinations
                data_dfA_unique = data_dfA.drop_duplicates().values.tolist()
                data_dfB_unique = data_dfB.drop_duplicates().values.tolist()
                
                # convert list of lists to list of tuples
                data_dfA_unique = [tuple(l) for l in data_dfA_unique]
                data_dfB_unique = [tuple(l) for l in data_dfB_unique]
            
                combos_A = set(data_dfA_unique)
                combos_B = set(data_dfB_unique)
                
                num_only_inA = len(combos_A - combos_B)
                num_only_inB = len(combos_B - combos_A)
                num_inboth = len(combos_A & combos_B)
                
                print('{} and {}:\t A - B: {} B - A: {} AuB: {}'.format(dset_nameA, dset_nameB, num_only_inA, num_only_inB, num_inboth))


if __name__ == '__main__':
    main()