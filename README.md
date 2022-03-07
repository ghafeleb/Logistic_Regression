# LR
## Model
In this code, I have provided learning Logistic Regression employing Minibatch SGD. To keep the code simple, I deleted Newton method. However, if you are interested, I can add it and share the code again.
To run the code, run the following command:
```bash
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/metr-la.h5
```
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
I use MNIST data as my test case. There are 25 pairs of odd-even pairs of digits in MNIST data  ((0, 1), (0, 2), …, (8, 9)) that can be used and they are indexed as follows:
- (0, 1): 0
- (0, 3): 1
- …
- (8, 9): 24
Test case is a binary classification of only one pair. To change the 
