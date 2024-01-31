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
The script generates a considerable volume of data and might take several minutes to finish...
2. To verify the data was successfully generated, view its statistics\
`python3 print_data_stats.py --db_dir PatientData`
3. Train on the generated, synthetic data:
   - Create the directory to store the results\
     `mkdir result_v0_syntDataCheck`
   - Run training:\
     `(export CUDA_VISIBLE_DEVICES=0 SPLIT=1; python3 train.py ${SPLIT} ./PatientData/scan_database.json ./PatientData/patient_database.json ./PatientData/split_map.json ./PatientData/ result_v0_syntDataCheck/log_split_${SPLIT})`
   - The training loss can be observed by plotting the log in file; with gnuplot:\
     `gnuplot -e "plot \"result_v0_syntDataCheck/log_split_1/log_train_basic.txt\" u 1; pause -1"`
   - Keep the training running for around 100 epochs to observe a decreasing trend in the plot;
   - Interrupt the training with `ctr+C`
4. Generate predictions:\
   `(export CUDA_VISIBLE_DEVICES=0 SPLIT=1; python3 predict_split.py ./PatientData/ ./PatientData/scan_database.json ./PatientData/split_map.json ${SPLIT} ./result_v0_test/log_split_${SPLIT}/net_last.pth ./result_v0_test/log_split_${SPLIT}/output_last.npy)`
5. Watch the resulting performance using the `analyseResults.ipynb` jupyter notebook
