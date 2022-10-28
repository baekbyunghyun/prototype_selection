import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.random_data_generator import RandomDataGenerator
from prototype_selector import PrototypeSelector


def get_number_of_prototypes(dataset=None, type_prototype_selection=None, tau=0.4, radius=0.2):
    selector = PrototypeSelector(algorithm=type_prototype_selection, X=dataset.X, y=dataset.y,
                                 parameter_tau=tau, parameter_radius=radius)
    X_prototype, y_prototype = selector.select()

    return len(X_prototype)


def draw_bar_graph_with_style(data_sizes, ips_num_of_prototypes, pbl_num_of_prototypes, hrps_num_of_prototypes, tau):
    indexes = np.arange(len(data_sizes))

    columns = ['Algorithm', 'Data Length', 'No. prototypes']

    data_frame = pd.DataFrame(columns=columns)

    for index, num_of_prototypes in zip(data_sizes, ips_num_of_prototypes):
        data_frame = data_frame.append(pd.DataFrame(data=[['IPS', index * 3, num_of_prototypes]], columns=columns))

    for index, num_of_prototypes in zip(data_sizes, pbl_num_of_prototypes):
        data_frame = data_frame.append(pd.DataFrame(data=[['PBL', index * 3, num_of_prototypes]], columns=columns))

    for index, num_of_prototypes in zip(data_sizes, hrps_num_of_prototypes):
        data_frame = data_frame.append(pd.DataFrame(data=[['IHRPS', index * 3, num_of_prototypes]], columns=columns))

    data_frame = data_frame.reset_index(drop=True)

    sns.set(style='darkgrid', color_codes=True)

    plt.figure(figsize=(10, 6))
    plt.title('The number of prototypes with random data', fontsize=16)

    ax = sns.barplot(x='Data Length', y='No. prototypes', hue='Algorithm', data=data_frame)

    hatches = itertools.cycle(['/////', '\\\\\\', 'xx', '//'])
    num_locations = len(indexes)

    for i, bar in enumerate(ax.patches):
        if i % num_locations == 0:
            hatch = next(hatches)

        bar.set_hatch(hatch)

    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.legend()
    plt.savefig('C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\number_of_prototypes_tau_{0}.PNG'.format(tau))
    # plt.show()
    plt.clf()


def draw_bar_graph(data_sizes, ips_num_of_prototypes, pbl_num_of_prototypes, hrps_num_of_prototypes, tau):
    indexes = np.arange(len(data_sizes))
    bar_width = 0.25
    opacity = 0.8

    plt.title('The number of prototypes with random data')

    plt.bar(indexes, ips_num_of_prototypes,
            alpha=opacity, color=(0.9, 0.9, 0.9), width=bar_width, hatch='/////', label='IPS', edgecolor='black', linewidth=1)
    plt.bar(indexes + bar_width, pbl_num_of_prototypes,
            alpha=opacity, color=(0.8, 0.8, 0.8), width=bar_width, hatch='\\\\\\', label='PBL', edgecolor='black', linewidth=1)
    plt.bar(indexes + (bar_width * 2), hrps_num_of_prototypes,
            alpha=opacity, color=(0.7, 0.7, 0.7), width=bar_width, hatch='//', label='HRPS', edgecolor='black', linewidth=1)

    plt.xlabel('Data size')
    plt.ylabel('No. prototypes')

    plt.tick_params(labelsize=9)
    plt.xticks(indexes + bar_width, data_sizes * 3)
    plt.grid(True, 'major', 'y', ls='--', lw=.1, c='k', alpha=1)
    plt.legend()

    plt.savefig('C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\number_of_prototypes_tau_{0}.PNG'.format(tau))
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    MIN_CLASS_DATA_SIZE = 100
    MAX_CLASS_DATA_SIZE = 1000
    STEP_DATA_SIZE = 100

    data_sizes = np.arange(MIN_CLASS_DATA_SIZE, MAX_CLASS_DATA_SIZE + 1, STEP_DATA_SIZE)

    tau_list = np.arange(0.3, 0.4, 0.1)
    for tau in tau_list:
        hrps_num_of_prototypes = list()
        pbl_num_of_prototypes = list()
        ips_num_of_prototypes = list()

        for size in data_sizes:
            generator = RandomDataGenerator(
                file_name='C:\\Users\\BAEK\\Desktop\\github\\HyperRectangle\\workspace\\dataset\\sample.spa',
                num_of_data_per_class=[size, size, size], col_size=2)
            generator.generate_2d_3c()

            hrps_num_of_prototype = get_number_of_prototypes(dataset=generator.dataset,
                                                            type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10,
                                                            tau=tau)
            hrps_num_of_prototypes.append(hrps_num_of_prototype)

            ips_num_of_prototype = get_number_of_prototypes(dataset=generator.dataset,
                                                        type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_IPS,
                                                        radius=0.1)
            ips_num_of_prototypes.append(ips_num_of_prototype)

            pbl_num_of_prototype = get_number_of_prototypes(dataset=generator.dataset,
                                                        type_prototype_selection=PrototypeSelector.TYPE_ALGORITHM_PBL)
            pbl_num_of_prototypes.append(pbl_num_of_prototype)

            print('Tau: {0}, Data size: {1}'.format(tau, size))
            print('HRPS: {0}, IPS: {1}, PBL: {2}'.format(hrps_num_of_prototype, ips_num_of_prototype, pbl_num_of_prototype))
            print()

        print('IPS: ', ips_num_of_prototypes)
        print('PBL: ', pbl_num_of_prototypes)
        print('HRPS: ', hrps_num_of_prototypes)

        # draw_bar_graph(data_sizes, ips_num_of_prototypes, pbl_num_of_prototypes, hrps_num_of_prototypes, tau)
        draw_bar_graph_with_style(data_sizes, ips_num_of_prototypes, pbl_num_of_prototypes, hrps_num_of_prototypes, tau)
