# Project: Image Classifier
The second project of [Udacity's](https://www.udacity.com) **Data Scientist Nanodegree**

## Motivation
Learn to use PyTorch, an open-source machine learning library, for image classification tasks. In this project, I'll build a image classifier to classify flower images up to 102 categories. The project templates are provided by [Udacity](https://www.udacity.com/)

## Project Files
* `Image Classifier Project.ipynb`: Notebook for dataset exploration and model building  
* `train.py`: Scripts for model training
* `predict.py`: Scripts for prediction

## Basic Usage
* Use command `python train.py data_directory` to train the model. This program will print out various loss and accuracy metrics as the network trains
    * `train.py` options: There are several options for this command
        * `--save_dir`: checkpoint saving directory
        * `--arch`: vgg model name (default: `vgg19_bn`)
        * `--learning_rate`: learning rate
        * `--first_hidden_units`: # of first hidden units
        * `--second_hidden_units`: # of second hidden units
        * `--epochs`: # of iteration
        * `--gpu`: use of gpu
* Use command `python predict.py /image_path checkpoint` to predict flower name with probability from an image 
    * `predict.py` options: 
        * `--top_k`: return top k results
        * `--category_name`: class name dictionary
        * `--gpu`: use of gpu

