import numpy as np
import scipy.io as sio
import svm

def first_dataset():
    data = sio.loadmat('dataset1.mat')
    y = data['y'].astype(np.float64)
    x = data['X']
    return x, y

def first_data_display():
    svm.visualize_boundary_linear(first_dataset()[0], first_dataset()[1], None, 'Исходные данные dataset1')

def learn_dataset1():
    model = svm.svm_train(first_dataset()[0], first_dataset()[1], 1, svm.linear_kernel, 0.001, 20)
    svm.visualize_boundary_linear(first_dataset()[0], first_dataset()[1], model, 'Разделяющая граница при C=1')

def main():
    first_data_display()
    learn_dataset1()

if __name__ == '__main__':
    main()