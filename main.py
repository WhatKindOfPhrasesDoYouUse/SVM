import numpy as np
import scipy.io as sio
import svm

def first_dataset():
    data = sio.loadmat('dataset1.mat')
    y = data['y'].astype(np.float64)
    x = data['X']
    return x, y

def second_dataset():
    data = sio.loadmat('dataset2.mat')
    y = data['y'].astype(np.float64)
    x = data['X']
    return x, y

def third_dataset():
    data = sio.loadmat('dataset3.mat')
    y = data['y'].astype(np.float64)
    x = data['X']
    x_val = data['Xval']
    y_val = data['yval'].astype(np.float64)
    return x, y, x_val, y_val

def first_data_display():
    svm.visualize_boundary_linear(first_dataset()[0], first_dataset()[1], None, 'Исходные данные dataset1')

def second_data_display():
    svm.visualize_boundary_linear(second_dataset()[0], second_dataset()[1], None, 'Исходные данные dataset2')

def third_data_display():
    svm.visualize_boundary_linear(third_dataset()[0], third_dataset()[1], None, 'Исходные данные обучающей выборки')
    svm.visualize_boundary_linear(third_dataset()[2], third_dataset()[3], None, 'Исходные данные тестовой выборки')

def learn_dataset1(C: int):
    model = svm.svm_train(first_dataset()[0], first_dataset()[1], C, svm.linear_kernel, 0.001, 20)
    svm.visualize_boundary_linear(first_dataset()[0], first_dataset()[1], model, 'Разделяющая граница при C={}'.format(C))

def gaussian_kernel_display(sigma: int):
    svm.contour(sigma)

def learn_dataset2(C, sigma):
    gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
    gaussian.__name__ = svm.gaussian_kernel.__name__
    model = svm.svm_train(second_dataset()[0], second_dataset()[1], C, gaussian)
    svm.visualize_boundary(second_dataset()[0], second_dataset()[1], model)

def suboptimal_learning_dataset3(C, sigma):
    gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
    gaussian.__name__ = svm.gaussian_kernel.__name__
    model = svm.svm_train(third_dataset()[0], third_dataset()[1], C, gaussian)
    svm.visualize_boundary(third_dataset()[0], third_dataset()[1], model)

def main():
    #first_data_display()
    #learn_dataset1(1)
    #learn_dataset1(100)
    #gaussian_kernel_display(1)
    #gaussian_kernel_display(3)
    #second_data_display()
    #learn_dataset2(1.0, 0.1)
    #third_data_display()
    #suboptimal_learning_dataset3(1, 0.5)

if __name__ == '__main__':
    main()