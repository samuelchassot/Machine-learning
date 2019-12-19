# BERT for Tweets Classification

## Depedencies
The notebook requires the following :
Either : 
- Google Colab
- ktrain (to install using the command !pip3 install ktrain)
Either :
- Python 3.8
- Jupyter Notebook
- Tensorflow 2.0 and all its dependencies or/and tensorflow-gpu depending on your computer
- Keras
- ktrain (to install using the command !pip3 install ktrain)

You can see this page https://www.tensorflow.org/install/gpu if you plan to do it locally and not in Google Colab.

The script requires :
- Python 3.8
- Tensorflow 2.0 and all its dependencies or/and tensorflow-gpu depending on your computer
- Keras
- ktrain (to install using the command !pip3 install ktrain)

The provided code is really time and resources consuming when done on a CPU so we strongly advice you to use Google Colab or to set up the code such as to use a good GPU.

## Datasets
Provided datasets must be extracted into a folder named `Datasets`, which then gives, for example, the following path for train_neg.txt : `Datasets/twitter-datasets/train_neg.txt`.

You should prepare a folder `BERT_folder` containing two sub-directories `train` and `test`, which both contain two sub-directories `pos` and `neg` before running the code.

## How to run ?
Either use the command python3 run.py, either you can import the notebook in Google Colab or run it using Jupyter Notebook. On Google Colab, you can use the function `text.texts_from_array` of the ktrain library if `text.texts_from_folder` creates problems due to the creation of the folder.

# Overview
This code demonstrates the use of the ktrain library to classify tweets into two categories, positive (1) and negative (-1). 
We split our training set into two subsets, one for training (75% of the original set) and for validation (25% of the orignal set).
We use a batch size of 10 to be scalable with most computer configurations.
We use a maximal number of tokens to consider (maxlen) of 199 since it is more scalable to compute and as seen manipulating the data for CNN, it is a plausible number for the maximal number of useful tokens to process in a tweet.
We use a learning rate of 2e-5 as suggested in BERT orignal paper and in the tutorial of ktrain on text classification.
We do 1 epoch since the computation is too long on a CPU to do more.
