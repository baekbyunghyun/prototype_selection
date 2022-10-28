import glob
import time
import os
import numpy as np

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from dataset.data_reader import DataReader
from prototype_selector import PrototypeSelector


def test_using_dt(dataset=None, type_prototype_selection=None, cv=5, radius=0.1, tau=0.1):
    result_num_of_prototypes = list()
    result_prototype_selection_times = list()
    result_learning_times = list()
    result_accuracy_trains = list()
    result_accuracy_tests = list()
    result_f1_trains = list()
    result_f1_tests = list()

    classifier = DecisionTreeClassifier(random_state=0)

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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

        else:
            selector = PrototypeSelector(algorithm=type_prototype_selection, X=X_train_data, y=y_train_data,
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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

    result_num_of_prototypes = np.array(result_num_of_prototypes)
    result_prototype_selection_times = np.array(result_prototype_selection_times)
    result_learning_times = np.array(result_learning_times)
    result_accuracy_trains = np.array(result_accuracy_trains)
    result_accuracy_tests = np.array(result_accuracy_tests)
    result_f1_trains = np.array(result_f1_trains)
    result_f1_tests = np.array(result_f1_tests)

    print('------------------------   INFORMATION   ------------------------')
    print('Decision Tree')
    print('tau: {0}'.format(tau))
    print('radius: {0}'.format(radius))
    print('[RESULT]')
    print('Number of prototypes:        {0}'.format(result_num_of_prototypes))
    print('Prototype selection times:   {0}'.format(result_prototype_selection_times))
    print('Times:                       {0}'.format(result_learning_times))
    print('Train accuracy:              {0}'.format(result_accuracy_trains))
    print('Test accuracy:               {0}'.format(result_accuracy_tests))
    print('Mean train accuracy:         {0}'.format(np.mean(result_accuracy_trains)))
    print('Mean test accuracy:          {0}'.format(np.mean(result_accuracy_tests)))
    print('SD train accuracy:           {0}'.format(np.std(result_accuracy_trains)))
    print('SD test accuracy:            {0}'.format(np.std(result_accuracy_tests)))
    print('Train f1 score:              {0}'.format(result_f1_trains))
    print('Test f1 score:               {0}'.format(result_f1_tests))
    print('Mean train f1 score:         {0}'.format(np.mean(result_f1_trains)))
    print('Mean test f1 score:          {0}'.format(np.mean(result_f1_tests)))
    print('-----------------------------------------------------------------')

    return np.mean(result_accuracy_trains), np.mean(result_accuracy_tests)


def test_using_svm(dataset=None, type_prototype_selection=None, cv=5, radius=0.1, tau=0.1):
    result_num_of_prototypes = list()
    result_prototype_selection_times = list()
    result_learning_times = list()
    result_accuracy_trains = list()
    result_accuracy_tests = list()
    result_f1_trains = list()
    result_f1_tests = list()

    classifier = SVC(kernel='rbf', C=10000.0, random_state=0, gamma='auto')

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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

        else:
            selector = PrototypeSelector(algorithm=type_prototype_selection, X=X_train_data, y=y_train_data,
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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

    result_num_of_prototypes = np.array(result_num_of_prototypes)
    result_prototype_selection_times = np.array(result_prototype_selection_times)
    result_learning_times = np.array(result_learning_times)
    result_accuracy_trains = np.array(result_accuracy_trains)
    result_accuracy_tests = np.array(result_accuracy_tests)
    result_f1_trains = np.array(result_f1_trains)
    result_f1_tests = np.array(result_f1_tests)

    print('------------------------   INFORMATION   ------------------------')
    print('Support Vector Machine')
    print('tau: {0}'.format(tau))
    print('radius: {0}'.format(radius))
    print('[RESULT]')
    print('Number of prototypes:        {0}'.format(result_num_of_prototypes))
    print('Prototype selection times:   {0}'.format(result_prototype_selection_times))
    print('Times:                       {0}'.format(result_learning_times))
    print('Train accuracy:              {0}'.format(result_accuracy_trains))
    print('Test accuracy:               {0}'.format(result_accuracy_tests))
    print('Mean train accuracy:         {0}'.format(np.mean(result_accuracy_trains)))
    print('Mean test accuracy:          {0}'.format(np.mean(result_accuracy_tests)))
    print('SD train accuracy:           {0}'.format(np.std(result_accuracy_trains)))
    print('SD test accuracy:            {0}'.format(np.std(result_accuracy_tests)))
    print('Train f1 score:              {0}'.format(result_f1_trains))
    print('Test f1 score:               {0}'.format(result_f1_tests))
    print('Mean train f1 score:         {0}'.format(np.mean(result_f1_trains)))
    print('Mean test f1 score:          {0}'.format(np.mean(result_f1_tests)))
    print('-----------------------------------------------------------------')

    return np.mean(result_accuracy_trains), np.mean(result_accuracy_tests)


def test_using_nn(dataset=None, type_prototype_selection=None, k=1, cv=5, tau=0.1, radius=0.1):
    result_num_of_prototypes = list()
    result_prototype_selection_times = list()
    result_learning_times = list()
    result_accuracy_trains = list()
    result_accuracy_tests = list()
    result_f1_trains = list()
    result_f1_tests = list()

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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

        else:
            selector = PrototypeSelector(algorithm=type_prototype_selection, X=X_train_data, y=y_train_data,
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
            result_f1_trains.append(f1_score(y_train_data, classifier.predict(X_train_data), average='micro'))
            result_f1_tests.append(f1_score(y_test_data, classifier.predict(X_test_data), average='micro'))

    result_num_of_prototypes = np.array(result_num_of_prototypes)
    result_prototype_selection_times = np.array(result_prototype_selection_times)
    result_learning_times = np.array(result_learning_times)
    result_accuracy_trains = np.array(result_accuracy_trains)
    result_accuracy_tests = np.array(result_accuracy_tests)
    result_f1_trains = np.array(result_f1_trains)
    result_f1_tests = np.array(result_f1_tests)

    print('------------------------   INFORMATION   ------------------------')
    print('Nearest neighbor with prototype')
    print('K:   {0}'.format(k))
    print('tau: {0}'.format(tau))
    print('radius: {0}'.format(radius))
    print('[RESULT]')
    print('Number of prototypes:        {0}'.format(result_num_of_prototypes))
    print('Prototype selection times:   {0}'.format(result_prototype_selection_times))
    print('Times:                       {0}'.format(result_learning_times))
    print('Train accuracy:              {0}'.format(result_accuracy_trains))
    print('Test accuracy:               {0}'.format(result_accuracy_tests))
    print('Mean train accuracy:         {0}'.format(np.mean(result_accuracy_trains)))
    print('Mean test accuracy:          {0}'.format(np.mean(result_accuracy_tests)))
    print('SD train accuracy:           {0}'.format(np.std(result_accuracy_trains)))
    print('SD test accuracy:            {0}'.format(np.std(result_accuracy_tests)))
    print('Train f1 score:              {0}'.format(result_f1_trains))
    print('Test f1 score:               {0}'.format(result_f1_tests))
    print('Mean train f1 score:         {0}'.format(np.mean(result_f1_trains)))
    print('Mean test f1 score:          {0}'.format(np.mean(result_f1_tests)))
    print('-----------------------------------------------------------------')

    return np.mean(result_accuracy_trains), np.mean(result_accuracy_tests)


if __name__ == '__main__':
    file_paths = glob.glob(os.path.join(r'C:\Users\BAEK\Desktop\research\HyperRectangle\uci_dataset', '*'))

    for file_path in file_paths:
        dataset = DataReader(file_path).get_dataset()
        dataset.scale()
        dataset.shuffle()

        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]

        print()
        print(file_name)
        print(dataset)

        test_using_dt(dataset=dataset, cv=3)
        test_using_nn(dataset=dataset, k=3, cv=3)
        #
        # test_using_dt(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_IPS, radius=0.05, cv=3)
        # test_using_nn(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_IPS, radius=0.05, k=3, cv=3)
        #
        # test_using_dt(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_PBL, cv=3)
        # test_using_nn(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_PBL, k=3, cv=3)
        #
        # tau_list = np.arange(0.1, 1.1, 0.1)
        # for tau in tau_list:
        #     # test_using_nn(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20, k=1, tau=tau, cv=3)
        #     test_using_dt(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20, tau=tau, cv=3)
        #
        # tau_list = np.arange(0.1, 1.1, 0.1)
        # for tau in tau_list:
        #     test_using_nn(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20, k=1, tau=tau, cv=3)
        #     # test_using_dt(dataset=dataset, type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20, tau=tau, cv=3)