# LR
## Model
The code is the implementation of Minibatch SGD to learn Logistic Regression model.
## Requirements
I have coded Logistic Regression in Python 3. The requirements are:
- numpy>=1.19.4
- torchvision>=0.3.0
- scikit-learn>=0.24.2
Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Data
I use MNIST data as my test case. There are 25 pairs of odd-even pairs of digits that can be used for our binary classification. Only one pair is used to train and test the model. To select the pair, pair_idx argument should be used. Indices of pairs are as follows:
- (0, 1): 0
- (0, 3): 1
- …
- (8, 9): 24

To be able to feed the data in Logistic Regression, the images are flattened. Moreover, PCA is used to reduce the size of the input data from 784 to 50. 
## Run the Model
To run the code, run the following command:
```bash
python -m LR_code --pair_idx=0 --batch_size=0 --n_epoch=25 --n_stepsize=10
```
where batch_size is the size of the batch in Minibatch SGD, n_epoch is the number of epochs, and n_stepsize is the number of stepsizes considered in hyperparameter tuning. Argument pair_idx is elaborated in Data subsection. Default value of batch_size is 0 which means the batch size is equal to size of the data. In other words, batch_size=0 means the algorithm is Gradient Descent.

