import pandas as pd
import os
import shutil
import numpy as np
import pydicom
from skimage.io import imsave
from random import sample
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# constants
boxes_path = 'maps/Annotation_Boxes.csv'
mapping_path = 'maps/Breast-Cancer-MRI-filepath_filename-mapping.csv'
data_path = None
target_png_dir = 'png_out'

def save_dcm_slice(dcm_fname, label, vol_idx, png_path=None):
    # label = 0 if negative/healthy
    # 1 if positive
    # print(dcm_fname)
    if png_path:
        pass
    else:
        png_path = dcm_fname.split('/')[-1].replace('.dcm', '-{}.png'.format(vol_idx))
        label_dir = 'pos' if label == 1 else 'neg'
        png_path = os.path.join(target_png_dir, label_dir, png_path)
    
    if not os.path.exists(png_path):
        try:
            dcm = pydicom.dcmread(dcm_fname)
        except FileNotFoundError:
            # fix by changing *-{abc}.dcm to *-{bc}.dcm
            dcm_fname_split = dcm_fname.split('/')
            dcm_fname_end = dcm_fname_split[-1]
            assert dcm_fname_end.split('-')[1][0] == '0'
            
            dcm_fname_end_split = dcm_fname_end.split('-')
            dcm_fname_end = '-'.join([dcm_fname_end_split[0], dcm_fname_end_split[1][1:]])
            
            dcm_fname_split[-1] = dcm_fname_end
            dcm_fname = '/'.join(dcm_fname_split)
            # dcm = pydicom.read_file(dcm_fname)
            dcm = pydicom.dcmread(dcm_fname)
            
        # way 1 (simple)
        # img = dcm.pixel_array.astype('float')
        
        # way 2
        # what we did for style transfer
        obj = dcm
        if obj.PixelSpacing[0] != obj.PixelSpacing[1]:
            raise UserWarning("Different spacing {} ".format(obj.PixelSpacing))
        img = obj.pixel_array
        img_type = obj.PhotometricInterpretation

        # uint16 -> float, scaled properly for uint8
        img = img.astype(np.float) * 255. / img.max()
        # float -> uint8
        img = img.astype(np.uint8)
        if img_type == "MONOCHROME1":
            img = np.invert(img)

        # print(dcm_arr.shape)
        imsave(png_path, img)


def make_png_subset():
    N_case = 20000 # make extra in case we want further experiments
    data_dir = 'png_out'
    target_dir = 'png_subset'
    
    for target_label in ['neg', 'pos']:
        case_dir = os.path.join(data_dir, target_label)
        
        sample_fnames = sample([fname for fname in os.listdir(case_dir) if '.png' in fname], N_case)
        for fname in tqdm(sample_fnames):
            fpath = os.path.join(case_dir, fname)

            # copy over
            img_path = os.path.join(target_dir, target_label, fname)
            # print(fpath, img_path)
            shutil.copy(fpath, img_path)

def main():
    if not os.path.exists(target_png_dir):
        os.makedirs(target_png_dir)

    # read boxes
    boxes_df = pd.read_csv(boxes_path)
    N_vols = len(boxes_df)
    print(N_vols)

    # read mapping 
    mapping_df = pd.read_csv(mapping_path)
    # in mapping csv, row order is:
    # { [post1 slices], ... [post_n slices],
    # pre fat-saturated slices (what we want),
    # T1 slices }

    # only keep rows that we need (fat-saturated "pre" exam only)
    mapping_df = mapping_df[mapping_df['original_path_and_filename'].str.contains('pre')]


    # main process
    # iter over each patient volume
    vol_idx = -1
    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    for row_idx, row in tqdm(mapping_df.iterrows(), total=mapping_df.shape[0]):
        # index start at 1 here
        new_vol_idx = int((row['original_path_and_filename'].split('/')[1]).split('_')[-1])
        slice_idx = int(((row['original_path_and_filename'].split('/')[-1]).split('_')[-1]).replace('.dcm', ''))
        
        # new volume: get boxes
        if new_vol_idx != vol_idx:
            box_row = boxes_df.iloc[[new_vol_idx-1]]
            start_slice = int(box_row['Start Slice'])
            end_slice = int(box_row['End Slice'])
            assert end_slice >= start_slice 
            # print(new_vol_idx, start_slice, end_slice) 
        vol_idx = new_vol_idx
        
        # print(slice_idx)
        # if row_idx > 10000:
        #     break
        
        # fix incorrect filenames
        dcm_fname = str(row['classic_path'])
        # dcm_fname_splt = dcm_fname.split('/')
        # wrong_entry = dcm_fname_splt[1]
        # dcm_fname_splt[1] = 'Breast_MRI_{}'.format(wrong_entry[-3:])
        # dcm_fname = '/'.join(dcm_fname_splt)
        
        # print(dcm_fname)
        dcm_fname = os.path.join(data_path, dcm_fname)
        # dcm_fname = dcm_fname.replace('
        # (1) if within 3D box, save as positive
        if slice_idx >= start_slice and slice_idx < end_slice: 
            # dcm = load_dcm(dcm_fname)
            save_dcm_slice(dcm_fname, 1, vol_idx)

        # (2) if outside 3D box by >5 slices, save as negative
        elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
            save_dcm_slice(dcm_fname, 0, vol_idx)


if __name__ == '__main__':
    main()

    # make subset for downstream usage
    make_png_subset()