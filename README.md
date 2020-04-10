#  Handwriting Recognition with Deep Learning implemented in TensorFlow.

Deep Learning System for the Recognition of Handwritten Words implemented in TensorFlow and trained with the IAM Handwriting Database.

A cross validation and the IAM test are performed on this system.


## Structure

### Python files:

- *clean_IAM.py*: Script for cleaning and preprocessing images..
- *ANN_model.py*: Neural network model implemented in TensorFlow.
- *cross-validation.py*: Script for cross validation.
- *train.py*: Script to train the model and store the parameters that achieve a better result.
- *test.py*: Script to test a previously trained model.
- *hw_utils.py*: Useful functions in different parts of the project.
- *csv_to_txt.py*: This file is for convert csv file to text file


### Requisitos Software
Python 3.6 y librerías:
- TensorFlow 1.3
- PIL
- Pandas
- Numpy
- Json
- Ast


### Installation and preprocessing of data.

After downloading or cloning the repository, it is necessary to download the dataset from the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and unzip it in the "Offline-Handwriting-Recognition-with-TensorFlow \ Data" directory.

Once we have obtained the dataset we execute:

```
python3 clean_IAM.py [path_config_file]
```
If no path is added to the configuration file the default path "./config.json" will be taken

This script selects the test-friendly images, resizes them, and adds padding to match their dimensions.
## Execution.

### Cross-validation.

To carry out cross-validation of the model, it is only necessary to execute:

```
python3 cross-validation.py [path_config_file]
```

This script performs 10 validations with different subdivisions of the original dataset and stores the results in CSV format.


### Test IAM

The first step is to train the model with the dataset offered by IAM with specific subdivisions. For this we execute:

```
python3 train.py [path_config_file]
```

This script performs a model training and stores the parameters that have given the best result for the validation dataset.

Once we have the model trained, we obtain the test result by executing:

```
python3 test.py [path_config_file]
```

The result is displayed on the screen and the system outputs are stored in CSV.

