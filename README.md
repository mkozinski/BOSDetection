Training a deep neural network for binary classification of CT scans.

# Prerequisites
1. For training and testing the neural network:
  - python3 (we used python 3.9.7),
  - numpy (we used 1.21.2),
  - pytorch (we used 1.11.0),
  - matplotlib for generating the final performance plots,
  - a program for plotting the loss functions during training; we used gnuplot;
2. To preprocess training and test scans that come in a dicom format, additionally:
  - scipy
  - pydicom
  - SimpleITK
  - lungmask from https://github.com/JoHof/lungmask 
3. Training is performed on a GPU and the training data is best stored on a fast SSD, since scans are read from the storage at each iteration

# Getting started
Test it on synthetic data (the instructions assume Linux/bash shell is used):
1. Checkout this repository and enter its directory\
   `git clone git@github.com:mkozinski/BOSDetection.git`\
   `cd BOSDetection`
2. Generate a synthetic data set\
`python3 generate_random_dataset.py`\
The script generates a considerable volume of data in the `PatientData` folder. It might take several minutes to finish...
2. To verify the data was successfully generated, view its statistics\
`python3 print_data_stats.py --db_dir PatientData`
3. Train on the generated, synthetic data:
   - Create the directory to store the results\
     `mkdir result_v0_syntDataCheck`
   - Run training:\
     `(export CUDA_VISIBLE_DEVICES=0 SPLIT=1; python3 train.py ${SPLIT} ./PatientData/scan_database.json ./PatientData/patient_database.json ./PatientData/split_map.json ./PatientData/ result_v0_syntDataCheck/log_split_${SPLIT})`
   - The training loss can be observed by plotting the log file; with gnuplot, this can be done as follows:\
     `gnuplot -e "plot \"result_v0_syntDataCheck/log_split_1/log_train_basic.txt\" u 1; pause -1"`
   - Keep the training running for around 100 epochs to observe a decreasing trend in the plot;
   - Interrupt the training with `ctr+C`
4. Generate predictions:\
   `(export CUDA_VISIBLE_DEVICES=0 SPLIT=1; python3 predict_split.py ./PatientData/ ./PatientData/scan_database.json ./PatientData/split_map.json ${SPLIT} ./result_v0_test/log_split_${SPLIT}/net_last.pth ./result_v0_test/log_split_${SPLIT}/output_last.npy)`
5. Watch the resulting performance using the `analyseResults.ipynb` jupyter notebook

# Training on own CT scans
We first describe the [data format](#data-format), then discuss [the script for converting dicom data into the desired format](#preparing-ct-scans-for-training), next introduce the [training script](#training-script) and the [prediction script](#prediction-script), and close this section with a reference to [analysing the results with the included juputer notebook](#analyse-results).

## Data format
The training/test data format is best inspected by viewing the synthetic data generated as described above. It comprises:
### The scan data base
   The scan data base for synthetic data is located in file `PatientData/scan_database.json`; Example entry looks like this:\
   ```
   "patient_19/scan_1995_12_9/scan_0159_copy_00.npy": {
        "date": {
            "day": 9,
            "month": 12,
            "year": 1995,
        },
        "patient": "patient_19",
        "scanner": "Scanner A",
    }
   ```
   It is a dictionary, where keys take the form of relative paths to files containing CT scans as numpy arrays. Absolute paths are created by concatenating these relative paths to the `root_dir` argument of multiple scripts in this repository. The values of dictionary entries are three-element dictionaries:
   -  the field "date" represents the date of scan acquisition;
   -  the field "patient" contains the patient identifier, a string;
   -  the field "scanner" contains the name of the device used to acquire the scan.
### The patient data base
    The patient data base for synthetic data is located in file `PatientData/patient_database.json`; Example entries look like this:\
    ```
    "patient_18": {
      "label": 0
    },
    "patient_19": {
      "FEV1_level_dates": {
          "0.5": {
              "day": 9,
              "month": 12,
              "year": 1995
          },
          "0.65": {
              "day": 23,
              "month": 7,
              "year": 1993
          },
          "0.8": {
              "day": 23,
              "month": 7,
              "year": 1993
          },
          "0.9": {
              "day": 27,
              "month": 3,
              "year": 1992
          },
          "start": {
              "day": 19,
              "month": 11,
              "year": 1990
          }
      },
      "label": 1
    }
    ```
   For the patient data base, keys take the form of patient identifiers. The values of dictionary entries are:
   -  the field "label"; 0&rarr; patient that did not develop the diseae; 1&rarr; patient that developed the disease, but early scans might show no symptoms;
   -  for the patients with "label"==1, a field called "FEV1_level_dates", containing the dates at which the patient's lung function decreased below pre-defined levels: 0.9, 0.8, 0.65, 0.5 of the "best value", taken to be the average of two best measurements taken at least 3 weeks apart. In order for the level to be effectively "crossed", the patient had to display a FEV1 measurement below this level on at least two consecutive scans taken at least two weeks apart. If the FEV1 decreased by more than one level, for example, from above 0.8 to below 0.65, the same date was assigned to both crossed levels, in the example, 0.8 and 0.65.
### The split map
  The split map for synthetic data is located in file `PatientData/split_map.json`; Example entries look like this:\
```
"patient_00": 0,
"patient_01": 1,
"patient_02": 0,
"patient_03": 1,
```
To each patient identifier, it assigns an integer that denotes the split in which the patient is assigned to the test set.
4. The scan files;\
They take the form of three-dimensional numpy arrays, where the dimensions are in the following order: Vertical (Z), Anterior-Posterior (Y), Medio-Lateral (X); This results in storing transverse slices of a scan in contiguous fragments of the file.

## Preparing CT scans for training
Scans in DICOM format can be converted to the desired format using the `preproc_scan.py` script. It is called as follows:\
`python3 preproc_scan.py --root_dir <root_dir> --outfile <out_file.npy> --scan_db <scan_database.json> --patient <patient_id> --infiles <infile_1.dicom> ... <infile_n.dicom>`\
where:\
   - `root_dir` is the root directory to be prepended to the local path of the output file `out_file`; this root part of the path is not stored in the `scan_database`
   - `outfile` is the local path to the output file in the numpy format
   - `scan_database` is the path to [the file that contains the information about individual scans](#the-scan-database); if the file exists, it will be updated; if the specific entry already exists, it will not be overwritten unless `--overwrite` has been specified
   - `patient_id` is the text identifier to be assigned to the patient; the same identifiers need to be used in [the patient data base file](#the-patient-database)
