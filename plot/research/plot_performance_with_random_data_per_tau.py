import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

from dataset.random_data_generator import RandomDataGenerator
from prototype_selector import PrototypeSelector


def test_for_nearest_neighbor(dataset=None, type_prototype_selection=None, k=1, cv=5, radius=0.1, tau=0.1):
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
                                         parameter_tau=tau, parameter_radius=radius)

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
    print('Nearest neighbor')
    print('K: ', k)
    print('tau: ', tau)
    print('radius: ', radius)
    print()
    print('[RESULT]')
    print('Number of prototypes: ', result_num_of_prototypes)
    print('Prototype selection times: ', result_prototype_selection_times)
    print('Times: ', result_learning_times)
    print('Train accuracy: ', result_accuracy_trains)
    print('Test accuracy: ', result_accuracy_tests)
    print('-----------------------------------------------------------------')

    return np.mean(result_accuracy_trains), np.mean(result_accuracy_tests)


def plot_accuracy(data_sizes,
                  results_train_from_3nn, results_test_from_3nn,
                  results_train_from_ips, results_test_from_ips,
                  results_train_from_pbl, results_test_from_pbl,
                  results_train_from_hrps, results_test_from_hrps, dir_path=None):
    plt.title('The comparison of train data accuracy using NN (k=3)')
    plt.plot(data_sizes * 3, results_train_from_3nn, marker='s', color='red', linewidth=1, linestyle=':', label='Without prototype selection')
    plt.plot(data_sizes * 3, results_train_from_ips, marker='o', color='blue', linewidth=1, linestyle=':', label='Hyper-sphere with fixed radius')
    plt.plot(data_sizes * 3, results_train_from_pbl, marker='x', color='magenta', linewidth=1, linestyle='--', label='Hyper-sphere with varialbe radius')
    plt.plot(data_sizes * 3, results_train_from_hrps, marker='d', color='green', linewidth=1, linestyle='-', label='Hyper-rectangle with variable diagonal')

    plt.ylim(0.0, 1.1)
    plt.xlabel('Data size')
    plt.ylabel('Accuracy')
    plt.grid(True, 'major', 'y', ls='--', lw=.1, c='k', alpha=1)
    plt.legend(loc='lower left')

    plt.savefig(os.path.join(dir_path, 'nn_with_random_train_data_tau2.PNG'))

    plt.clf()

    plt.title('The comparison of test data accuracy using NN (k=3)')
    plt.plot(data_sizes * 3, results_test_from_3nn, marker='s', color='red', linewidth=1, linestyle=':', label='Without prototype selection')
    plt.plot(data_sizes * 3, results_test_from_ips, marker='o', color='blue', linewidth=1, linestyle=':', label='Hyper-sphere with fixed radius')
    plt.plot(data_sizes * 3, results_test_from_pbl, marker='x', color='magenta', linewidth=1, linestyle='--', label='Hyper-sphere with varialbe radius')
    plt.plot(data_sizes * 3, results_test_from_hrps, marker='d', color='green', linewidth=1, linestyle='-', label='Hyper-rectangle with variable diagonal')

    plt.ylim(0.0, 1.1)
    plt.tick_params(labelsize=8)
    plt.xlabel('Data size')
    plt.ylabel('Accuracy')
    plt.grid(True, 'major', 'y', ls='--', lw=.1, c='k', alpha=1)
    plt.legend(loc='lower left')

    plt.savefig(os.path.join(dir_path, 'nn_with_random_test_data_tau2.PNG'))

    plt.clf()


if __name__ == '__main__':
    MIN_DATA_SIZE = 100
    MAX_DATA_SIZE = 1000
    STEP_DATA_SIZE = 100

    results_train_3nn_tau_01 = list()
    results_train_3nn_tau_02 = list()
    results_train_3nn_tau_03 = list()
    results_train_3nn_tau_04 = list()
    results_train_3nn_tau_05 = list()
    results_train_3nn_tau_06 = list()
    results_train_3nn_tau_07 = list()
    results_train_3nn_tau_08 = list()
    results_train_3nn_tau_09 = list()
    results_train_3nn_tau_10 = list()

    results_test_3nn_tau_01 = list()
    results_test_3nn_tau_02 = list()
    results_test_3nn_tau_03 = list()
    results_test_3nn_tau_04 = list()
    results_test_3nn_tau_05 = list()
    results_test_3nn_tau_06 = list()
    results_test_3nn_tau_07 = list()
    results_test_3nn_tau_08 = list()
    results_test_3nn_tau_09 = list()
    results_test_3nn_tau_10 = list()

    data_sizes = np.arange(MIN_DATA_SIZE, MAX_DATA_SIZE + 1, STEP_DATA_SIZE)
    tau_list = np.arange(0.1, 1.1, 0.1)

    for size in data_sizes:
        generator = RandomDataGenerator(
            file_name='C:\\Users\\BAEK\\Desktop\\github\\HyperRectangle\\workspace\\dataset\\sample.spa',
            num_of_data_per_class=[size, size, size], col_size=2)
        generator.generate_2d_3c()

        dataset = generator.dataset
        dataset.shuffle()
        dataset.scale()

        print(dataset)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.1, cv=10)
        results_train_3nn_tau_01.append(train_accuracy)
        results_test_3nn_tau_01.append(test_accuracy)


        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.2, cv=10)
        results_train_3nn_tau_02.append(train_accuracy)
        results_test_3nn_tau_02.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.3, cv=10)
        results_train_3nn_tau_03.append(train_accuracy)
        results_test_3nn_tau_03.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.4, cv=10)
        results_train_3nn_tau_04.append(train_accuracy)
        results_test_3nn_tau_04.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.5, cv=10)
        results_train_3nn_tau_05.append(train_accuracy)
        results_test_3nn_tau_05.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.6, cv=10)
        results_train_3nn_tau_06.append(train_accuracy)
        results_test_3nn_tau_06.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.7, cv=10)
        results_train_3nn_tau_07.append(train_accuracy)
        results_test_3nn_tau_07.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.8, cv=10)
        results_train_3nn_tau_08.append(train_accuracy)
        results_test_3nn_tau_08.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=0.9, cv=10)
        results_train_3nn_tau_09.append(train_accuracy)
        results_test_3nn_tau_09.append(test_accuracy)

        train_accuracy, test_accuracy = test_for_nearest_neighbor(dataset=generator.dataset,
                                                                  type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                                  k=3, tau=1.0, cv=10)
        results_train_3nn_tau_10.append(train_accuracy)
        results_test_3nn_tau_10.append(test_accuracy)

    print('3nn train accuracy   (tau=0.1):    {0}'.format(results_train_3nn_tau_01))
    print('3nn test accuracy    (tau=0.1):    {0}'.format(results_test_3nn_tau_01))
    print('3nn train accuracy   (tau=0.2):    {0}'.format(results_train_3nn_tau_02))
    print('3nn test accuracy    (tau=0.2):    {0}'.format(results_test_3nn_tau_02))
    print('3nn train accuracy   (tau=0.3):    {0}'.format(results_train_3nn_tau_03))
    print('3nn test accuracy    (tau=0.3):    {0}'.format(results_test_3nn_tau_03))
    print('3nn train accuracy   (tau=0.4):    {0}'.format(results_train_3nn_tau_04))
    print('3nn test accuracy    (tau=0.4):    {0}'.format(results_test_3nn_tau_04))
    print('3nn train accuracy   (tau=0.5):    {0}'.format(results_train_3nn_tau_05))
    print('3nn test accuracy    (tau=0.5):    {0}'.format(results_test_3nn_tau_05))
    print('3nn train accuracy   (tau=0.6):    {0}'.format(results_train_3nn_tau_06))
    print('3nn test accuracy    (tau=0.6):    {0}'.format(results_test_3nn_tau_06))
    print('3nn train accuracy   (tau=0.7):    {0}'.format(results_train_3nn_tau_07))
    print('3nn test accuracy    (tau=0.7):    {0}'.format(results_test_3nn_tau_07))
    print('3nn train accuracy   (tau=0.8):    {0}'.format(results_train_3nn_tau_08))
    print('3nn test accuracy    (tau=0.8):    {0}'.format(results_test_3nn_tau_08))
    print('3nn train accuracy   (tau=0.9):    {0}'.format(results_train_3nn_tau_09))
    print('3nn test accuracy    (tau=0.9):    {0}'.format(results_test_3nn_tau_09))
    print('3nn train accuracy   (tau=1.0):    {0}'.format(results_train_3nn_tau_10))
    print('3nn test accuracy    (tau=1.0):    {0}'.format(results_test_3nn_tau_10))


