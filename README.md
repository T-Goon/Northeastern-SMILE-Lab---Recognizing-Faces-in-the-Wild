# Northeastern SMILE Lab - Recognizing Faces in the Wild
A Kaggle competition done for the final project of WPI class CS 4342 Machine Learning.
Kaggle Competition: https://www.kaggle.com/c/recognizing-faces-in-the-wild/overview/description


Details on the assignment can be found in "final_project.pdf".

## Data Preprocessing

We took 224 x 224 RBG images of faces and created 128 dimension face emeddings (https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) with a pretrained facenet model (https://github.com/nyoki-mtl/keras-facenet). 

The input to our machine learning models predicted based on two emeddings as input. Therefore, we had to append these two embeddings before feeding them into the models. To promote symmetrical weights within our models, if we append embedding A onto embedding B we also made sure to include a sample in the training data where embedding B was appended onto embedding A.

An amount of random noise was also added to the face embeddings to promote a more generalized neural network model.

## Model

The model we used with the best results is a 4 layer dense neural network.

Layers:
1. Input Layer
2. Dense Layer 1 (15 neurons) with 20% dropout
3. Dense Layer 2 (15 neurons)
4. Softmax Activation Output Layer (2 neurons)

## Results

### Hyperparameter Optimization

After a grid search of hyperparameter combinations these ones turned out to be the best:
- Epochs: 20
- Batch Size: 64
- Number of Neurons in Both Hidden Layers: 15
- Learning Rate: .0001
- Activation Function: Scaled Exponential Linear
- Standard Deviation of Random Noise: .5

| Cross Entropy Loss  | Accuracy | Area Under ROC Curve |
| ------------------- | -------- | -------------------- |
|          .83        | .61      |         .70          |

A detailed report on the development process and results can be found in "final_project_jjlaforest_twgoon_omthode.pdf".

## Usage

To run: `python final_project.py`

## Dependencies

- Numpy 1.19.5
- Pandas 1.0.5
- Pillow 7.0.0
- Tensorflow 2.4.0
- Python 3.7.4
- Sklearn 0.0
- Matplotlib 3.1.2
