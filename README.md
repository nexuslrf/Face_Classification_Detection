# Face_Classification_Detection

## Intro

This project focuses on implementation and analysis of some classic machine learning models applied in human face classification tasks. The models I implemented include logistic regression, linear discriminant analysis(LDA), support vector machine(SVM), convolutional neural network(CNN). I make some comparisons of these models on the performance of face classification. Besides, a series of feature/representation analysis is done in order to get deep insights of how these machine learning models work. Finally, I implement a face detection framework which uses above models as back end, which shows the promising results.

## Environments

* python3.6 based jupyter notebook.
* NumPy
* Scikit-image + Scikit-learn
* Pytorch1.0

## Dataset and Checkpoints

* [FDDB database](<http://vis-www.cs.umass.edu/fddb/>): after downloading and unzipping  the dataset to the current root directory. Simply follow steps in [`DataProcess.ipynb`](./DataProcess.ipynb), you can get the processed data for our model training. 
* HOG features: after running through [`DataProcess.ipynb`](./DataProcess.ipynb), open  [`HOG.ipynb`](./HOG.ipynb),  you can see some visualization of HOG features and generate packed training & testing set for traditional ML models.
* The pretrained checkpoints for each model by myself are uploaded to my [OneDrive](https://1drv.ms/f/s!AtiMpA7HPe0Qhc4HEz3fBbVwfXgKoQ)

## Play with Model training & testing

Ipython notebooks with titles of specific models show the real implementation of each ML model. [./Model_zoo](./Model_zoo) collects each ML models in each separated `*.ipynb` .

To run training or testing  simple run through the  [`Training.ipynb`](./Training.ipynb) cell by cell, all codes here are very easy to understand.

## Play with Detection

Simply open  [`Detection.ipynb`](./Detection.ipynb), set the index, backend model and other related parameters, then you can make it work.

#### Demo(+1s):

<img src="det_demo.jpg" width="400" align=center />



 

