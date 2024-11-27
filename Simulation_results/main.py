import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from collections import Counter

def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}


import matlab.engine
from imblearn.datasets import make_imbalance

class HDDT:
    """
    A class to represent a HDDT classifier.
    
    ...

    Attributes
    ----------
    par : int
        just some random no.

    Methods
    -------
    HDDT(X_train,X_test,y_train1,y_test1):
        calculates classifier's to be predicted.
    """
    def __init__(self):
        self.ownclass = True

    def predict(self,x_train,x_test,y_train1):
        """
        calculates classifier's to be predicted.

        Parameters
        ----------
        x_train : matrix
            training samples
        x_test : matrix 
            testing samples
        y_train1 : matrix
            trainig labels
        Returns
        -------
        y_computed: matrix
            It returns computed test labels
        """
        # Start the MATLAB Engine
        eng = matlab.engine.start_matlab()
        # Convert Python arrays to MATLAB arrays
        x_train_matlab = matlab.double(x_train.tolist())
        x_test_matlab = matlab.double(x_test.tolist())
        y_train_matlab = matlab.double(y_train1.tolist())
     
        
        # Call the MATLAB function with the MATLAB arrays
        y_predict = eng.d_tree(x_train_matlab,x_test_matlab,y_train_matlab)
        
        eng.quit()
        return y_predict

class HDRF:
    """
    A class to represent a HDRF classifier.
    
    ...

    Attributes
    ----------
    par : int
        just some random no.

    Methods
    -------
    HDRF(X_train,X_test,y_train1):
        calculates classifier's to be predicted.
    """
    def __init__(self):
        self.ownclass = True

    def predict(self,x_train,x_test,y_train1):
        """
        calculates classifier's to be predicted.

        Parameters
        ----------
        x_train : matrix
            training samples
        x_test : matrix 
            testing samples
        y_train1 : matrix
            trainig labels
        Returns
        -------
        y_computed: matrix
            It returns computed test labels
        """
        # Start the MATLAB Engine
        eng = matlab.engine.start_matlab()
        # Convert Python arrays to MATLAB arrays
        x_train_matlab = matlab.double(x_train.tolist())
        x_test_matlab = matlab.double(x_test.tolist())
        y_train_matlab = matlab.double(y_train1.tolist())
        
        # Call the MATLAB function with the MATLAB arrays
        y_predict = eng.d_forest(x_train_matlab, x_test_matlab, y_train_matlab)
        eng.quit()
        return y_predict
class PNN:
    """
    A class to represent a PNN classifier.
    
    ...

    Attributes
    ----------
    par : int
        just some random no.

    Methods
    -------
    PNN(X_train,X_test,y_train):
        calculates classifier's to be predicted.
    """
    def __init__(self):
        self.ownclass = True

    def predict(self,x_train,x_test,y_train1):
        """
        calculates classifier's to be predicted.

        Parameters
        ----------
        x_train : matrix
            training samples
        x_test : matrix 
            testing samples
        y_train : matrix
            trainig labels
        Returns
        -------
        y_computed: matrix
            It returns computed test labels
        """
        # Start the MATLAB Engine
        eng = matlab.engine.start_matlab()
        # Convert Python arrays to MATLAB arrays
        x_train_matlab = matlab.double(x_train.tolist())
        x_test_matlab = matlab.double(x_test.tolist())
        y_train_matlab = matlab.double(y_train1.tolist())
        
        # Call the MATLAB function with the MATLAB arrays
        y_predict = eng.PNN_classifier(x_train_matlab, x_test_matlab, y_train_matlab)
        
        eng.quit()
        return y_predict

class SPNN:
    """
    A class to represent a SPNN classifier.
    
    ...

    Attributes
    ----------
    par : int
        just some random no.

    Methods
    -------
    BASPNN(X_train,X_test,y_train):
        calculates classifier's to be predicted.
    """
    def __init__(self):
        self.ownclass = True

    def predict(self,x_train,x_test,y_train1):
        """
        calculates classifier's to be predicted.

        Parameters
        ----------
        x_train : matrix
            training samples
        x_test : matrix 
            testing samples
        y_train : matrix
            trainig labels
        Returns
        -------
        y_computed: matrix
            It returns computed test labels
        """
        # Start the MATLAB Engine
        eng = matlab.engine.start_matlab()
        # Convert Python arrays to MATLAB arrays
        x_train_matlab = matlab.double(x_train.tolist())
        x_test_matlab = matlab.double(x_test.tolist())
        y_train_matlab = matlab.double(y_train1.tolist())
        
        # Call the MATLAB function with the MATLAB arrays
        y_predict = eng.BASPNN_classifier(x_train_matlab, x_test_matlab, y_train_matlab)
        
        eng.quit()
        return y_predict

h = .01  # step size in the mesh

names = ["HDDT", "PNNs", "SkewPNNs"]
classifiers = [
    HDDT(),
    PNN(),
    SPNN(),
]


# Define parameters for the half moons
n_points = 357  # Number of points for each half-moon
noise = 0.29   # Noise level to add variability

# Generate the first half-moon (Class 0)
theta1 = np.linspace(0, np.pi, n_points)  # Angles for the semi-circle
X1 = np.column_stack((np.cos(theta1), np.sin(theta1)))  # Coordinates
X1 += np.random.normal(scale=noise, size=X1.shape)  # Add noise
y1 = np.zeros(n_points, dtype=int)  # Label 0 for Class 0

# Generate the second half-moon (Class 1)
theta2 = np.linspace(0, np.pi, n_points)  # Angles for the semi-circle
X2 = np.column_stack((1 - np.cos(theta2), -np.sin(theta2)))  # Coordinates, shifted and flipped
X2 += np.random.normal(scale=noise, size=X2.shape)  # Add noise
y2 = np.ones(n_points, dtype=int)  # Label 1 for Class 1

# Combine the datasets and labels
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))


# Add noise to the data
noise_level = 0.27  # Adjust the noise level as needed
X += np.random.normal(0, noise_level, X.shape)

imbalance_ratios = [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]

figure = plt.figure(figsize=(15, 10))
i = 1

# iterate over imbalance ratios
for imbalance_ratio in imbalance_ratios:
    X_resampled, y_resampled = make_imbalance(
        X,
        y,
        sampling_strategy=ratio_func(y,0.3,1),
        **{"multiplier": imbalance_ratio[0], "minority_class": 1},
    )
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    cm = ListedColormap(['#A0CFC1','#F7FBC1'])
    cm_bright = ListedColormap(['#FFDB58','#0000FF'])
    ax = plt.subplot(len(imbalance_ratios), len(classifiers) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='black', cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='black', cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    temp_ratio = imbalance_ratio[0] * 100
    # Add label for imbalance ratio
    ax.text(xx.max() - 2.5 , yy.max() + 0.4, f'IR: {imbalance_ratio[0]:.2f}%',
        size=12, horizontalalignment='right', verticalalignment='top')

    i += 1

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(imbalance_ratios), len(classifiers) + 1, i)
        
        if hasattr(clf, "ownclass"):
            y_pred_matlab = clf.predict(X_train, X_test, y_train)
            y_pred_numpy = np.array(y_pred_matlab)
            auc_roc = roc_auc_score(y_test, y_pred_numpy)
            grid = np.c_[xx.ravel(), yy.ravel()]
            # print(grid.shape)
            y_pred_matlab = clf.predict(X_train, grid, y_train)
            y_pred_numpy = np.array(y_pred_matlab)
            Z = y_pred_numpy.reshape(xx.shape) 
            
            if name == 'PNN' or name == 'BA-SPNN':
                # Convert the NumPy matrix to a pandas DataFrame
                df = pd.DataFrame(Z)

                # Specify the file name
                file_name = "matrix_" + name + "_ibr_"+ str(int(imbalance_ratio[0]*100)) +".csv"

                # Save the pandas DataFrame to the CSV file
                df.to_csv(file_name, index=False, header=False)

        else:

            y_pred = clf.fit(X_train, y_train)
            # Get predicted probabilities for the positive class
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba)

            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='black', cmap=cm_bright)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='black', cmap=cm_bright,alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % auc_roc).lstrip('0'),
                size=15, horizontalalignment='right')
        
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()
