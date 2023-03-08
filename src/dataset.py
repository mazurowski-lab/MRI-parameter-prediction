import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

# constants
data_dirs = {
    'dbc_by_scanner' : 'data/dbc/png_subset/sorted_by_scanner',
}

cur_dir = os.getcwd()

class MedicalDataset(Dataset):
    def __init__(self, label_csv, data_dir, img_size, transform, make_3_channel=False, return_filenames=False):
        self.label_csv = label_csv
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        self.make_3_channel = make_3_channel
        
        # to be initialized by child class
        self.labels = None
        self.filenames = None

        #options
        self.return_filenames = return_filenames
                 
    def normalize(self, img):
        # normalize to range [0, 255]
        # img expected to be array
                 
        # uint16 -> float
        img = img.astype(float) * 255. / img.max()
        # float -> unit8
        img = img.astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        
        fpath, target  = self.labels[idx]
        
        # load img from file (png or jpg)
        img_arr = io.imread(fpath, as_gray=True)
        
        # normalize
        img_arr = self.normalize(img_arr)
        
        # convert to tensor
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 
       
        # add channel dim
        data = torch.unsqueeze(data, 0)
        
        # resize to standard dimensionality
        data = transforms.Resize((self.img_size, self.img_size))(data)
        # bilinear by default
        
        # make 3-channel (testing only)
        if self.make_3_channel:
            data = torch.cat([data, data, data], dim=0)
        
        # do any data augmentation/training transformations
        if self.transform:
            data = self.transform(data)

        ret = [data, target]
        if self.return_filenames:
            ret.append(self.filenames[idx])

        if self.return_filepaths:
            ret.append(fpath)
        
        return ret
    
    def __len__(self):
        return len(self.labels)
    
class DBCDataset(MedicalDataset):
    def __init__(self, img_size, labeling='feature: TE', train_transform=None, make_3_channel=False, unique_patients=False, return_filenames=False, num_classes=None, task='classification', test_this_IAP_only=None,
    different_cases_for_train_val_test=False, return_filepaths=False, regress_dense_IAPs=False):
        super(DBCDataset, self).__init__(None, data_dirs['dbc_by_scanner'], img_size, train_transform, make_3_channel=make_3_channel, return_filenames=return_filenames)
        # constants
        self.clinical_features_path = 'data/dbc/maps/Clinical_and_Other_Features.csv'
        self.return_filepaths = return_filepaths

        # with number of classes, or 1 if regression
        self.all_feature_names = OrderedDict([
            #contrast bolus volume and reconstruction diameter not included due to a decent amt of missing vals
           ('Manufacturer', 3),
           ('Manufacturer Model Name' , 8),
           ('Scan Options' , 9),
           ('Patient Position During MRI' , 2),
           ('Field Strength (Tesla)' , 4),
           ('Contrast Agent' , 6),
           ('Acquisition Matrix', 10),
           ('Slice Thickness ', 21),
           ('Flip Angle \n', 4),
           ('FOV Computed (Field of View) in cm ', 27),
           ('TE (Echo Time)', 1),
           ('TR (Repetition Time)', 1)
        ])

        if test_this_IAP_only is not None:
            print('Using only patients with ', test_this_IAP_only)


        dense_IAPs = [
            'Slice Thickness ',
            'Flip Angle \n', 
            'FOV Computed (Field of View) in cm '
            ]
        clinical_features_cats_path = 'data/dbc/maps/clinical_feature_categories.csv'
        clinical_features_cats = pd.read_csv(clinical_features_cats_path)
        if regress_dense_IAPs:
            for feature in dense_IAPs:
                self.all_feature_names[feature] = 1

        labels = []
        filenames = []
        patient_IDs_used = []
        # (fname, value = label (0 = neg, 1 = pos) )
        # print('building DBC dataset.')
        if labeling == 'default':
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        patient_ID = fname.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)
                        
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
                        filenames.append(file)
        elif labeling == 'all':
            # load features
            clinical_features = pd.read_csv(self.clinical_features_path)

            # loop through scans
            missing_features_count = 0
            missing_feature_patients = []
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".png"):
                        fpath = os.path.join(root, file)
                        patient_ID = file.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)

                        patient_ID = 'Breast_MRI_{}'.format(patient_ID.zfill(3))
                        # get data for named features
                        all_feature_vals = {}

                        has_missing_feature = False
                        exclude_this_scan = False
                        for feature_name in self.all_feature_names.keys():
                            feature_val = clinical_features[clinical_features['Patient ID'] == patient_ID][feature_name].values[0]

                            if regress_dense_IAPs and (feature_name in dense_IAPs):
                                # possibly regress certain categorical IAPs that have many categories
                                # convert a predicted category index for an IAP into its value
                                feature_pred = int(feature_val)
                                feature_map = clinical_features_cats[feature_name].iloc[0]
                                feature_map = feature_map.split(',')
                                feature_map = [x.strip() for x in feature_map]
                                feature_map = {int(x.split('=')[1]): x.split('=')[0] for x in feature_map}
                                feature_val = feature_map[feature_pred] 

                            if test_this_IAP_only:
                                target_IAP_name = list(test_this_IAP_only.keys())[0]
                                if feature_name == target_IAP_name and feature_val != test_this_IAP_only[feature_name]:
                                    exclude_this_scan = True


                            try:
                                # if feature is for regression
                                if self.all_feature_names[feature_name] == 1:
                                    all_feature_vals[feature_name] = float(feature_val)
                                else:
                                    #classification
                                    all_feature_vals[feature_name] = int(feature_val)
                            except:
                                #print('missing feature: ', feature_name, ' for patient: ', patient_ID)
                                has_missing_feature = True

                        if has_missing_feature:
                            # if feature is missing/NaN, skip this patient
                            if patient_ID not in missing_feature_patients:
                                missing_feature_patients.append(patient_ID)
                                missing_features_count += 1
                            continue

                        if exclude_this_scan:
                            continue

                        filenames.append(file)
                        labels.append((fpath, all_feature_vals))
            
            print('there are missing features for {} patients'.format(missing_features_count))

        elif 'feature' in labeling:
            # load features
            clinical_features = pd.read_csv(self.clinical_features_path)


            feature_name = labeling.split(': ')[1]#.strip()
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".png"):
                        fpath = os.path.join(root, file)
                        patient_ID = file.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)

                        # get data for named feature
                        patient_ID = 'Breast_MRI_{}'.format(patient_ID.zfill(3))
                        feature_val = clinical_features[clinical_features['Patient ID'] == patient_ID][feature_name].values[0]

                        try:
                            if task == 'classification':
                                feature_val = int(feature_val)
                            elif task == 'regression':
                                feature_val = float(feature_val)
                            else:
                                raise NotImplementedError

                            labels.append((fpath, feature_val))
                            filenames.append(file)
                        except:
                            continue

        else:
            raise NotImplementedError

        if num_classes:
            raise NotImplementedError

        self.filenames = filenames
        self.labels = labels

    def sample_one_label_only(self, this_label_only):
        # sample dataset for one label only
        new_filenames = []
        new_labels = []
        print(len(self.labels))
        for idx, label in enumerate(self.labels):
            if label[1] == this_label_only:
                new_filenames.append(self.filenames[idx])
                new_labels.append(label)

        self.filenames = new_filenames
        self.labels = new_labels
        print(len(self.labels))

    def get_cancer_labels(self, filenames):
        # get cancer labels for batch of filenames
        cancer_labels = torch.zeros(len(filenames))
        
        for i, filename in enumerate(filenames):
            if os.path.exists(os.path.join('data/dbc/png_subset/sorted_by_cancer/pos', filename)):
                cancer_labels[i] = 1
        return cancer_labels.long()
        # 0 = neg 1 = pos

         
# utils
def get_different_cases_for_train_val_test(dataset, train_size, val_size, test_size):
    train_indices, val_indices, test_indices = [], [], []
    loader = DataLoader(dataset, batch_size=1)
    
    # get all patient ids
    unique_patient_ids = []
    for data_idx, batch in enumerate(loader):
        filename = batch[2]
        filename = filename[0]
        patient_ID = filename.split('-')[2].replace('.png', '')
        if patient_ID not in unique_patient_ids:
            unique_patient_ids.append(patient_ID)

    # split patient ids into train, val, test according to desired ratio from subset sizes 
    print('number of unique patients: {}'.format(len(unique_patient_ids)))
    num_patients_train = int(len(unique_patient_ids) * train_size/len(dataset))
    num_patients_val = int(len(unique_patient_ids) - num_patients_train) // 2
    num_patients_test = int(len(unique_patient_ids) - num_patients_train - num_patients_val)
    assert num_patients_train + num_patients_val + num_patients_test == len(unique_patient_ids)

    # split patient ids into train, val, test
    train_patient_ids, eval_patient_ids = train_test_split(unique_patient_ids, train_size=num_patients_train, test_size=num_patients_val+num_patients_test)
    val_patient_ids, test_patient_ids = train_test_split(eval_patient_ids, train_size=num_patients_val, test_size=num_patients_test)

    # get indices of all images from each patient id
    for data_idx, batch in enumerate(loader):
        assert len(batch) >= 3
        filename = batch[2]

        filename = filename[0]
        patient_ID = filename.split('-')[2].replace('.png', '')
        if patient_ID in train_patient_ids:
            train_indices.append(data_idx)
        elif patient_ID in val_patient_ids:
            val_indices.append(data_idx)
        elif patient_ID in test_patient_ids:
            test_indices.append(data_idx)

    # print('number of train, val, test images after splitting by patient: {}, {}, {}'.format(len(train_indices), len(val_indices), len(test_indices)))
    return train_indices, val_indices, test_indices
    


def get_datasets(dataset_name, labeling='default', train_size=None, 
                 test_size=None, val_size=None, img_size=224, make_3_channel=False,
                 unique_DBC_patients=False, return_filenames=False, num_classes=None, task="classification",
                 test_this_IAP_only=None, different_cases_for_train_val_test=False,
                 return_filepaths=False, regress_dense_IAPs=False):
    # either (1) specify train_frac, which split of subset to create train and test sets, or
    # (2) specify test_size
    
    if labeling != 'default':
        print('using non-default labeling: {}'.format(labeling))

    # first, option of getting subset of full dataset stored
    # then, option of splitting what's left into train and test
    # create dataset
    if dataset_name == 'dbc_by_scanner':
        dataset = DBCDataset(img_size, labeling, make_3_channel=make_3_channel, unique_patients=unique_DBC_patients, return_filenames=return_filenames, num_classes=num_classes, task=task, test_this_IAP_only=test_this_IAP_only, different_cases_for_train_val_test=different_cases_for_train_val_test,
        return_filepaths=return_filepaths, regress_dense_IAPs=regress_dense_IAPs)
    else:
        raise NotImplementedError
        
    # split into subsets if chosen
    if train_size and val_size and test_size:
        if train_size + val_size + test_size < len(dataset):
            dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), train_size + val_size + test_size, replace=False))

        if different_cases_for_train_val_test:
            # get indices for cases from different volumes for the train, val and test sets
            # dataset sizes will not be exactly as specified, due to constraints of having different cases
            train_indices, val_indices, test_indices = get_different_cases_for_train_val_test(dataset, train_size, val_size, test_size)
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

        else: # split randomly
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1337))


        return train_dataset, val_dataset, test_dataset
    else:
        return dataset