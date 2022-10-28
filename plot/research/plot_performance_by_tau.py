import time
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

from dataset.data_reader import DataReader
from prototype_selector import PrototypeSelector


def test_for_nearest_neighbor(dataset=None, type_prototype_selection=None, k=1, tau=0.1, cv=5):
    assert dataset is not None, "Invalid arguments"

    result_num_of_prototypes = list()
    result_prototype_selection_times = list()
    result_learning_times = list()
    result_accuracy_trains = list()
    result_accuracy_tests = list()

    classifier = neighbors.KNeighborsClassifier(k, weights='uniform', algorithm='brute', p=2)

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=20)
    for idx_train, idx_test in kf.split(dataset.X, dataset.y):
        X_train_data = dataset.X[idx_train]
        y_train_data = dataset.y[idx_train]
        X_test_data = dataset.X[idx_test]
        y_test_data = dataset.y[idx_test]

        if type_prototype_selection is None:
            start_time = time.time()

            classifier.fit(X_train_data, y_train_data)

            elapsed = time.time() - start_time
            result_learning_times.append(elapsed)

            result_accuracy_trains.append(classifier.score(X_train_data, y_train_data))
            result_accuracy_tests.append(classifier.score(X_test_data, y_test_data))

        else:
            selector = PrototypeSelector(algorithm=type_prototype_selection, X=dataset.X, y=dataset.y,
                                         parameter_tau=tau)

            start_time = time.time()

            X_prototype, y_prototype = selector.select()

            elapsed = time.time() - start_time
            result_prototype_selection_times.append(elapsed)

            result_num_of_prototypes.append(len(X_prototype))

            start_time = time.time()

            classifier.fit(X_prototype, y_prototype)

            elapsed = time.time() - start_time
            result_learning_times.append(elapsed)

            result_accuracy_trains.append(classifier.score(X_train_data, y_train_data))
            result_accuracy_tests.append(classifier.score(X_test_data, y_test_data))

    result_num_of_prototypes = np.array(result_num_of_prototypes)
    result_prototype_selection_times = np.array(result_prototype_selection_times)
    result_learning_times = np.array(result_learning_times)
    result_accuracy_trains = np.array(result_accuracy_trains)
    result_accuracy_tests = np.array(result_accuracy_tests)

    print('------------------------   INFORMATION   ------------------------')
    print('Nearest neighbor with prototype')
    print('K: ', k)
    print('tau: ', tau)
    print()
    print('[RESULT]')
    print('Number of prototypes: ', result_num_of_prototypes)
    print('Prototype selection times: ', result_prototype_selection_times)
    print('Times: ', result_learning_times)
    print('Train accuracy: ', result_accuracy_trains)
    print('Test accuracy: ', result_accuracy_tests)
    print('-----------------------------------------------------------------')

    return np.mean(result_accuracy_trains), np.mean(result_accuracy_tests)


def plot_accuracy(tau_list, train_accuracies_without_prototypes, test_accuracies_without_prototypes,
                  train_accuracies_using_prototypes, test_accuracies_using_prototypes, file_path=None):
    # plt.figure(figsize=(6, 10))
    plt.title('Performance nearest neighbor (k=3)')

    plt.plot(tau_list, train_accuracies_without_prototypes, marker='', color='red', linewidth=2, label='Train accuracy without prototypes')
    plt.plot(tau_list, test_accuracies_without_prototypes, marker='', color='blue', linewidth=2, label='Test accuracy without prototypes')
    plt.plot(tau_list, train_accuracies_using_prototypes, marker='o', color='green', linewidth=3, linestyle='dashed', label='Train accuracy with prototypes')
    plt.plot(tau_list, test_accuracies_using_prototypes, marker='x', color='lightblue', linewidth=2, linestyle='dashed', label='Test accuracy with prototypes')

    plt.ylim(0.1, 1.1)
    plt.xlabel('tau')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(loc='lower left')

    plt.savefig(file_path)
    plt.clf()
    # plt.show()


def test_for_target(dataset, save_file_name):
    train_accuracies_without_prototypes = list()
    test_accuracies_without_prototypes = list()
    train_accuracies_using_prototypes = list()
    test_accuracies_using_prototypes = list()

    tau_list = np.arange(0.1, 1.1, 0.1)
    for t in tau_list:
        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3,
                                                                  tau=t,
                                                                  cv=3)

        train_accuracies_using_prototypes.append(train_accuracy)
        test_accuracies_using_prototypes.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=dataset,
                                                                  k=3,
                                                                  cv=3)

        train_accuracies_without_prototypes.append(train_accuracy)
        test_accuracies_without_prototypes.append(test_accuracy)

    print("-------- without prototype -----------")
    print(train_accuracies_without_prototypes)
    print(test_accuracies_without_prototypes)

    print("---------- using prototype -----------")
    print(train_accuracies_using_prototypes)
    print(test_accuracies_using_prototypes)

    plot_accuracy(tau_list,
                  train_accuracies_without_prototypes,
                  test_accuracies_without_prototypes,
                  train_accuracies_using_prototypes,
                  test_accuracies_using_prototypes,
                  file_path='C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\performance_{0}.PNG'.format(save_file_name))


if __name__ == '__main__':
    file_paths = glob.glob('C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\dataset\\*')

    for file_path in file_paths:
        dataset = DataReader(file_path).get_dataset()
        dataset.scale()
        dataset.shuffle()

        print('------------------------------------------------------------------')
        print('[DEBUG]')
        print(file_path)
        print(dataset)

        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]

        test_for_target(dataset, file_name)

        print('------------------------------------------------------------------')
        print()



