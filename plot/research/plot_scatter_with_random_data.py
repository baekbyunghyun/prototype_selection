from dataset.data_reader import DataReader
from dataset.random_data_generator import RandomDataGenerator
from plot.plot_instance import PlotInstance
from plot.plot_manager import PlotManager
from prototype_selector import PrototypeSelector
from ds_paper.dsplot import *


def plot_scatter_for_fixed_radius(dataset, file_path, radius=0.1):
    selector = PrototypeSelector(algorithm='IPS', X=dataset.X, y=dataset.y, parameter_radius=radius)
    X_prototype, y_prototype = selector.select()

    print('Number of prototypes using fixed radius: ' + str(len(X_prototype)))

    plot_instance = PlotInstance(X=dataset.X, y=dataset.y)
    plt = PlotManager.plot_with_radius(title='Prototype selection based on IPS',
                                       plot_instance=plot_instance, X_prototype=X_prototype,
                                       y_prototype=y_prototype, ep=radius)
    plt.savefig(file_path)


def plot_scatter_for_hyper_rectange(dataset, file_path, tau=0.3):
    selector = PrototypeSelector(algorithm=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_10, X=dataset.X, y=dataset.y, parameter_tau=tau)
    X_prototype, y_prototype = selector.select()

    print('Number of prototypes using hyper rectangle: ' + str(len(X_prototype)))

    hr_list = selector.get_hyper_rectangles()

    plot_instance = PlotInstance(X=dataset.X, y=dataset.y)
    plt = PlotManager.plot_with_rectangles(title='Prototype selection based on HRPS',
                                           plot_instance=plot_instance, Hr=hr_list)
    plt.savefig(file_path)


def plot_scatter_for_variable_radius(dataset, file_path, tau=0, theta=0.3):
    selector = PrototypeSelector(algorithm=PrototypeSelector.TYPE_ALGORITHM_PBL, X=dataset.X, y=dataset.y,
                                 parameter_tau=tau)
    X_prototype, y_prototype = selector.select()

    print('Number of prototypes using hyper sphere: ' + str(len(X_prototype)))

    plot_instance = PlotInstance(dataset.X, dataset.y)
    plt = PlotManager.plot_with_radii(title='Prototype selection based on PBL',
                                      plot_instance=plot_instance,
                                      prototype_indexes=selector.get_prototype_index(),
                                      radiuses=selector.get_hyper_sphere_radius())

    plt.savefig(file_path)


def plot_scatter_for_hyper_rectangle_prototype(dataset, file_path, tau=0.3):
    selector = PrototypeSelector(algorithm=PrototypeSelector.TYPE_ALGORITHM_HRPS_V_20, X=dataset.X, y=dataset.y, parameter_tau=tau)
    X_prototype, y_prototype = selector.select()

    print('Number of prototypes using hyper rectangle: ' + str(len(X_prototype)))

    hr_list = selector.get_hyper_rectangles()

    plot_instance = PlotInstances(dataset.X, dataset.y)
    plt = PlotManager.plot_prototypes_with_rectangles(title='Prototype selection based on hyper-rectangles', plot_instance=plot_instance, Hr=hr_list)
    plt.savefig(file_path)


if __name__ == '__main__':
    generator = RandomDataGenerator(
        file_name='C:\\Users\\BAEK\\Desktop\\github\\HyperRectangle\\workspace\\dataset\\sample.spa',
        num_of_data_per_class=[300, 300, 300], col_size=2)
    generator.generate_2d_3c()

    # data_reader = DataReader('C:\\Users\\BAEK\\Desktop\\github\\HyperRectangle\\workspace\\dataset\\fourclass_scale.spa')
    data_reader = DataReader('C:\\Users\\BAEK\\Desktop\\github\\HyperRectangle\\workspace\\dataset\\sample.spa')
    dataset = data_reader.get_dataset()
    dataset.shuffle()
    dataset.scale()

    print(dataset)

    plot_scatter_for_hyper_rectange(dataset, 'C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\hyper_rectange_with_HRPS.PNG', tau=0.4)
    plot_scatter_for_variable_radius(dataset, 'C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\hyper_sphere_with_PBL.PNG')
    plot_scatter_for_fixed_radius(dataset, 'C:\\Users\\BAEK\\Desktop\\research\\HyperRectangle\\image\\hyper_sphere_with_IPS.PNG', radius=0.1)

