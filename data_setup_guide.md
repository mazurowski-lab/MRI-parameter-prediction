# Breast MRI dataset setup tutorial for the results of "Reverse Engineering Breast MRIs: Predicting Acquisition Parameters Directly from Images"

In this document we detail how to set up the data used in this paper, in order to reproduce our results.

### (1) Download the Duke Breast Cancer MRI data
This paper involves training models on seven different radiological image datasets. Here we detail how to set up these datasets for further experiments.

1. Download the Duke-Breast-Cancer-MRI dataset from The Cancer Imaging Archive [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). This will take a couple of steps; all files are found under "Data Access" near the bottom of the page.

2. First, download the annotation and filepath list files "File Path mapping tables (XLSX, 49.6 MB)" and "Annotation Boxes (XLSX, 49 kB)" into `data/dbc/maps`. You'll then need to convert these to `.csvs` manually (e.g. using Microsoft Excel).

3. Download the DBC Dataset "Images (DICOM, 368.4 GB)" as follows; unfortunately this is large due to the data only being avaliable as DICOM files. You'll have to use TCIA's NBIA Data Retriever tool for this (open the downloaded `.tcia` manifest file with the tool). **Make sure that you download the files with the "Classic Directory Name" directory type convention.** Otherwise certain files will be mislabeled in the downloaded annotation file, and you'll have to redownload all data from scratch. There are still certain typos in the downloaded annotation files from TCIA, but the subsequent code that we provide has fixes for these.

4. Once all of the data is downloaded, it will be in a folder named `manifest-{...}`, where `{...}` is some auotgenerated string of numbers, for example `manifest-1607053360376`. This folder may be within a subdirectory or two. Move this manifest folder into `data/dbc`.

5. Open the IPython notebook `data/dbc/png_extractor.ipynb`; in cell 2, modify `data_path` in line 4 to be equation to the name of your manifest folder, `manifest-{...}`.

6. Run all cells of this IPython notebook to extract .png images from the raw DICOM files into `data/dbc/png_out`.

7. To create the subset of images that we'll use for experiments, and sort them by scanner manufacturer, run the IPython notebook `data/dbc/make_subset.ipynb`.

8. Feel free to delete the original DICOM files in your manifest folder once this is complete, as well as `data/dbc/png_out`, to save space.