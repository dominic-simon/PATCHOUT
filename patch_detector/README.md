# Patch Detector Training

This directory contains a few pre-trained patch detectors (```ResNet50_ap.pt```, ```ResNet50_gdpa.pt```, and ```ResNet50_lavan.pt```) and scripts to train your own. 

## Dataset Creation

The dataset class is contained in ```csv_dataset.py```. A custom dataset for training can be created with the ```create_dataset.py``` file. Prior to creating the dataset, you will need to generate images attacked by adversarial patches. You will need to fill in the path to your attacked images on line 32 and the path to the corresponding benign versions of the images on line 35. To be clear, for each image you want to add to the dataset, you need an attacked version and a benign version.

The ```create_dataset.py``` file has 4 command line arguments:
- --csv_name: The file path the dataset will be saved as. Do not include a file extension. The file will automatically be saved as a CSV.
- --num_images: The number of images to include in the dataset.
- --use_gpu: Include this flag if you would like to use a GPU for computations.
- --gpu_num: The GPU number that will be used if you are using a GPU.

For example:

``` commandline 
python3 create_dataset.py --csv_name toy_dataset --num_images 100 --use_gpu --gpu_num 0
```

Or, more realistically: 

```commandline
python3 create_dataset.py --csv_name real_dataset --num_images 10000 --use_gpu --gpu_num 0
```

It is recommended that you create both a training and testing dataset. 

## Patch Detector Training

The patch detector model class is provided in ```patch_detector.py```. Use the ```train.py``` script to train patch detectors with the dataset you created above. 

The ```train.py``` file has 12 command line arguments:
- --save_model_name: The file path for your model after training.
- --test_model_name: The file path of a previously trained model to be used on a testing dataset.
- --train_dataset_name: The file path of the training dataset.
- --test_dataset_name: The file path of the testing dataset.
- --batch_size: The batch size to be used during training.
- --lr: The learning rate to be used during training.
- --momentum: The momentum to be used with optimization.
- --num_epochs: The number of epochs during training.
- --do_train: Set this flag to train a new model. If this flag is not set, no training will occur.
- --do_test: Set this flag to test an existing model. If this is not set, no testing will occur.
- --use_gpu: Include this flag if you would like to use a GPU for computations.
- --gpu_num: The GPU number that will be used if you are using a GPU.

Do not add any file extensions (```.pt``` or ```.csv```) to any file paths. The file extensions are automatically added in the code.

For example: 

```commandline
python3 train.py --save_model_name example_model --test_model_name example_model --train_dataset_name training_ds --test_dataset_name testing_ds --batch_size 32 --lr 1e-4 --momentum 0.9 --num_epochs 10 --do_train --do_test --use_gpu --gpu_num 0
```
Please refer to the paper (Table 2, page 8) for parameters on datasets and patches that have results shown in the paper. If you are trying a different dataset or patch, you will need to spend some time configuring the parameters to find what works best. 
