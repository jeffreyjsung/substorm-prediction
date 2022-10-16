# Transfer Learning Aurora Image Classification and Magnetic Disturbance Evaluation (TAME)
In our publication we show that transfer learning can be easily applied to all sky images for the purpose of classifying images, filtering images and predicting magnetic disturbance from auroral images.
In the publication we describe our methods and show the results that we obtained.  
Our data is archived on the NIRD Research Data Archive. It is licensed under CC-BY 4.0 and available for at least 10 years after publication. The archive with accompanying information can be accessed here:  [https://doi.org/10.11582/2021.00057](https://doi.org/10.11582/2021.00057)  
On this website we will
1. provide and describe the code we used in our publication for replication of our results.
2. provide instructions on how to apply the classifier in 6 lines of code.
3. provide a way to conglomerate different kinds of data used in space physics in order to be able to perform large scale data analysis using our code.

## 1. Replication of Results & Quickstart
For a full documentation of this package see the points below.
1. Make sure all dependencies are installed:
   ```bash
   sudo apt-get install python3 wget p7zip
   ```
2. Download our code:
   ```bash
   cd <Your working directory here>
   wget http://tid.uio.no/TAME/data/code.7z
   7z x code.7z
   cd code
   ```
3. Create a conda environment from the environment file we provide. This may take a few moments. Afterwards, activate the environment, to use it. You can find installation instructions for conda here:[https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
   ```bash
   conda env create -f environment.yml
   conda activate dataHandler
   ```
4. Create a storage folder, download our fully preprocessed data archives and extract them into the just created folder:
   ```bash
   mkdir /home/data
   mkdir /home/data/images
   wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/torch_data.7z -P /home/data
   wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/other_data.7z -P /home/data
   7z x /home/data/torch_data.7z -o/home/data
   7z x /home/data/other_data.7z -o/home/data
   ```
5. Run Python, import our Package and plot the results:
   ```bash
   python
   ```
   ```python
   from dataHandler import DataAnalyzer
   other_data_location = "/home/data/other/"
   torch_data_location = "/home/data/torch/"
   image_location = "/home/data/images/"
   da = DataAnalyzer(analyzer_data_path=other_data_location, torch_path=torch_data_location, image_path=image_location)
   da.plot_daily_cloud_coverage()
   da.plot_conf_images()
   da.plot_cloud_predictions_roc()
   da.demonstrate_cluster()
   da.plot_magn_predictions()
   ```
   This will create the figures we show in our publication and save them in /home/data/images.

## 2. Usage of the ASIM Classifier
Follow steps 1 to 3 of the tutorial above to set up the necessary environment.
Next, download and extract the classifier:
```bash
mkdir /home/data
mkdir /home/data/torch
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/oath_clf.7z
7z x oath_clf.7z -o/home/data/torch/
```
In a python shell you can access the classifier the following way:
```python
import glob
from dataHandler import AsimClassifier
torch_data_location = "/home/data/torch/"
clf = AsimClassifier(torch_path=torch_data_location)
image_file_list = glob.glob("path/to/asim/files/*.png")
results = clf.classify_images(image_file_list)
```
The default batch size will be 64, if you want to increase it, you can pass the parameter "batch_size" to `clf.classify_images()`.
Because the neural network we use employs batch normalization, we recommend not to go below the current default batch size.

## 3. Documentation
### Retrieval of data
We use all sky imager data that is freely available through [UiOs Resources](https://tid.uio.no/plasma/aurora/).
Please see [here](http://tid.uio.no/plasma/aurora/usage.html) if you want to use this data.
However, the total file size of all images taken in the timeframe Nov 2010 - Feb 2011 is about 200GB, which is why we kindly ask you to use the preprocessed files that we provide below.
If you are interested in the original image files, please let us know, and we will arrange for a transfer.
We provide the following archives for preprocessed files:
 - [asim_data.7z](https://doi.org/10.11582/2021.00057) This archive contains the processed all sky imager data files.
   The files' default location is `data/asim`
 - [magnetometer_data](https://space.fmi.fi/image/www/index.php?page=home) is available through the International Monitor for Auroral Geomagnetic Effects (IMAGE).
   Data can be downloaded from their website and preprocessed using our preprocessing library.
   The magnetometer data's default location is `data/magn`
 - [ceilometer_data](https://doi.pangaea.de/10.1594/PANGAEA.880300) is provided by the Alfred Wegener Institute through PANGAEA.
   Downloaded files can be processed with our preprocessor.
   The ceilometer data's default location is `data/ceil`
 - [oath_data](http://tid.uio.no/plasma/oath/) is provided by UiO.
   After Download and extraction, these files can be used to extract the OATH-images' features and train the classifier using the preprocessing library.
   The OATH data's default location is `data/oath`
 - [oath_clf.7z](https://doi.org/10.11582/2021.00057) This archive contains only the classifier obtained from the OATH images.
   The classifier's default location is `data/torch/oath.clf`.
 - [torch_data.7z](https://doi.org/10.11582/2021.00057) This archive contains other meta-data obtained to test the different neural networks for classification.
   The files' default location is `data/torch`.
 - [other_data.7z](https://doi.org/10.11582/2021.00057) This archive contains the highest level of processed data.
   The files contained herein are at the last processed stage of our code and are what is used to create the results and figures we present in our publication.
   This folder's default location is `data/other`
   
Due to the individual size of the archives, we provide them separately.
```bash
mkdir ~/data
mkdir ~/data/torch
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/asim_data.7z -P ~/data
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/oath_clf.7z -P ~/data/torch
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00057/other_data.7z -P ~/data
7z x ~/data/asim_data.7z -o/home/data
7z x ~/data/oath_clf.7z -o/home/data/torch
7z x ~/data/other_data.7z -o/home/data
```

The preprocessed files are always of the same structure:
Data is bundled into hdf-files per month, named YYYYMM.hdf.
The hdf-files contain a pandas dataframe, where the first columns contain information about time and place the data was taken, followed by columns containing the data, one row per entry.
The format of the first columns is the same for every data type we provide to make cross-referencing between them easier.

### Usage
Please see the first three steps under "Replication of Results & Quickstart" for how to set up the code.
Our package is structured in several classes abstracted from the main class `DataHandler`.

#### Preprocessor
The `PreProcessor` is processing the downloaded data and converts them into the hdf-files we provide in the archives above.
This is an example for how to preprocess the all sky imager data and set a custom path for the folder in which to look for the data and to store the preprocessed data in.
Similar functions and parameters exist for all other data types.

```python
from dataHandler import PreProcessor
asim_path = "own_folder/asim_data/"
pp = PreProcessor(ceil_path=asim_path)
pp.proc_asim(batch_size=64)
```
Contrary to the other functions that mostly only transcribe the data into easier to read storage in pandas dataframes, when processing the all sky imager data, they are run through a convolutional neural network that extracts each image's features, which are then saved per image.
Instead of having to use about 200GB of image data for the 2010/11 season, we have reduced the total size to about 6.2GB.

Furthermore the PreProcessor can `evaluate_network_performances` and `evaluate_network_accuracies` to benchmark the performance of different pretrained neural network architectures against the OATH Images, find the best `svm_hyper_parameters` by performing a gridsearch and finally `fit_oath_features` to create a classifier based in the OATH images that is able to `predict_image_proba` of the six classes for any given all sky image, if the features have been extracted by the same neural network.
In order to extract the features, we provide functions to `set_model_and_device` the same way we did as well as a dataset class that is compatible with pytorch's dataloader.
It can be imported and used as
```python
from dataHandler.datasets import AsimDataSet
from torch.utils.data import DataLoader
file_list = []
index_list = []
data = AsimDataSet(file_list, index_list)
dl = DataLoader(data, shuffle=False, batch_size=64)
for i_batch, sample in enumerate(dl, 0):
    pass
```
Here, `file_list` is a list of the image files and `index_list` ist a list of unique, numerical indices used to address these files.

#### Provider
The `Provider` provides data for a given timeframe and location.
This is an example for how to retrieve ceilometer and all sky imager data taken in Ny Ålesund between the 1st and 4th of December 2010.
```python
from dataHandler import Provider
from datetime import datetime
pr = Provider()
date_start = datetime(2010,12,1)
date_end = datetime(2010,12,4)
location = "NYA"
ceil_data = pr.get_ceil(date_start=date_start, date_end=date_end, location=location)
asim_data = pr.get_asim(date_start=date_start, date_end=date_end, location=location)
```
Because we want to compare different types of data, we provide utility to combine two sets of data.
Here, data from the second set of data is combined into the first set of data, such that the maximum time difference between the combined points is as low as possible.
If for a point of data in the first set no point of data in the second set within a timeframe of 86400s can be found, the point is discarded.
The column of the second dataframe that is to be merged into the first dataframe has to be provided as an argument of the merging function.
Due to the nature of this operation that necessitates comparing sometimes tens of thousands of rows for as many times, this might take a while.
Since we only expect data in the way we intended the tool to be used for, this function splits any input data on into manageable daily chunks.
This means that around midnight points might be merged, where the nearest point might have been on the next or previous day, but the current day has been chosen.
Compared to the amount of data, we judged this to be acceptable considering the speed-up of the merging gained by this.
```python
asim_and_ceil = pr.combine_data_sets(asim_data, ceil_data, "CBH")
```
#### Analyzer
The `DataAnalyzer` is the class that performs the operations that use the processed data, combines them and analyzes and presents them.
The same way as described above this can be used to create all the figures that we show in the publication
```python
from dataHandler import DataAnalyzer
other_data_location = "/home/data/other_data"
oath_data_location = "/home/data/oath_data"
image_location = "/home/data/images"
da = DataAnalyzer(analyzer_data_path=other_data_location, oath_path=oath_data_location, image_path=image_location)
da.plot_daily_cloud_coverage()
da.plot_conf_images()
da.plot_cloud_predictions_roc()
da.demonstrate_cluster()
da.plot_magn_predictions()
```

## Questions
If you have any questions or remarks, please [send me an e-mail](mailto:pascal.sado@fys.uio.no).

## References
The data is archived here:  
[https://doi.org/10.11582/2021.00057](https://doi.org/10.11582/2021.00057)  
If you have not already done so, please read our publication based on this data:  
[https://doi.org/10.1002/essoar.10507386.1](https://doi.org/10.1002/essoar.10507386.1)

If you use our classifier, this library in general or our publication, you can cite us the following way:  
```
@article{10.1002/essoar.10507386.1,
   author = {Sado, Pascal and Clausen, Lasse Boy Novock and Miloch, Wojciech Jacek and Nickisch, Hannes},
   title = {Transfer Learning Aurora Image Classification and Magnetic Disturbance Evaluation},
   journal = {Earth and Space Science Open Archive},
   pages = {26},
   year = {2021},
   DOI = {10.1002/essoar.10507386.1},
   url = {https://doi.org/10.1002/essoar.10507386.1},
}
```

## Acknowledgements and Copyright
The source code in this library is licensed under a [BSD-2-Clause License](https://opensource.org/licenses/BSD-2-Clause).
Unless stated otherwise, all data contained in the datasets we provide ourselves alongside this publication under the links above are licensed under a [Creative Commons Attribution-NonCommercial 4.0 License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The copyright for the all-sky imager data, some data-files are derived from, remains with the original copyright holder, the University of Oslo.
Information on how to use the original image-files can be obtained here:  
http://tid.uio.no/plasma/aurora/usage.html

