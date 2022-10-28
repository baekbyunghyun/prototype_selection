import numpy as np
import matplotlib.pyplot as plt


class PlotManager:
    @staticmethod
    def init_outline():
        plt.clf()

        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

    @staticmethod
    def plot_with_raw_data(plot_instance, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        for ea in range(plot_instance.K):
            idx = np.where(plot_instance.y == ea)[0]

            plt.plot(plot_instance.X[idx, 0], plot_instance.X[idx, 1], marker[ea], markersize=3)

        plt.axis(plot_instance.axis)

        return plt

    @staticmethod
    def plot_with_centers(plot_instance, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        for ea in range(plot_instance.K):
            idx = np.where(plot_instance.y == ea)[0]
            cen = np.mean(plot_instance.X[idx], axis=0)

            plt.plot(plot_instance.X[idx, 0], plot_instance.X[idx, 1], marker[ea])
            plt.plot(cen[0], cen[1], "kD", markersize=5)

            nc = plt.Circle(cen, 1, color='b', clip_on=True, fill=False)
            fig = plt.gcf()
            fig.gca().add_artist(nc)

        plt.axis(plot_instance.axis)

        return plt

    @staticmethod
    def plot_with_radius(title='', plot_instance=None, X_prototype=None, y_prototype=None, ep=None, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        for ea in np.arange(plot_instance.K):
            idx = np.where(plot_instance.y == ea)[0]
            plt.plot(plot_instance.X[idx, 0], plot_instance.X[idx, 1], marker[ea], markersize=1)

        fig = plt.gcf()

        for x_proto, y_proto in zip(X_prototype, y_prototype):
            nc = plt.Circle(x_proto, ep, color='b', clip_on=True, fill=False)
            fig.gca().add_artist(nc)

            plt.plot(x_proto[0], x_proto[1], 'x', markersize=6, color=marker[y_proto][0], markeredgewidth=2)

        plt.axis(plot_instance.axis)
        plt.title(title)

        return plt

    @staticmethod
    def plot_with_radii(title='', plot_instance=None, prototype_indexes=None, radiuses=None, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        for ea in np.arange(plot_instance.K):
            idx = np.where(plot_instance.y == ea)[0]
            plt.plot(plot_instance.X[idx, 0], plot_instance.X[idx, 1], marker[ea], markersize=1)

        fig = plt.gcf()

        for prototype_index in prototype_indexes:
            circle = plt.Circle(plot_instance.X[prototype_index], radiuses[prototype_index],
                                color='blue', clip_on=True, fill=False)
            fig.gca().add_artist(circle)

            plt.plot(plot_instance.X[prototype_index][0], plot_instance.X[prototype_index][1], 'x',
                     markersize=6, markeredgewidth=2, color=marker[plot_instance.y[prototype_index]][0])

        # for x_proto, y_proto in zip(X_prototype, y_prototype):
        #     nc = plt.Circle(x_proto, ep, color='b', clip_on=True, fill=False)
        #     fig.gca().add_artist(nc)
        #
        #     plt.contour(x_proto[0], x_proto[1], 'x', markersize=6, color=marker[y_proto][0], markeredgewidth=2)

        plt.axis(plot_instance.axis)
        plt.title(title)

        return plt

    @staticmethod
    def plot_with_rectangles(title='', plot_instance=None, Hr=None, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        for ea in np.arange(plot_instance.K):
            idx = np.where(plot_instance.y == ea)[0]
            plt.plot(plot_instance.X[idx, 0], plot_instance.X[idx, 1], marker[ea], markersize=1)

        fig = plt.gcf()

        for hr in Hr:
            idx = hr.y
            (width, height) = hr.x_max - hr.x_min

            nc = plt.Rectangle(hr.x_min, width, height, color='b', clip_on=True, fill=False)
            fig.gca().add_artist(nc)

            plt.plot(hr.x_mid[0], hr.x_mid[1], 'x', markersize=6, color=marker[idx][0], markeredgewidth=2)

        plt.title(title)
        plt.axis(plot_instance.axis)

        return plt

    @staticmethod
    def plot_prototypes_with_rectangles(title='', plot_instance=None, Hr=None, subfig='111'):
        PlotManager.init_outline()
        marker = ["ro", "k+", "bx", "ko", "b+", "rx"]

        plt.subplot(subfig)

        fig = plt.gcf()

        for hr in Hr:
            idx = hr.y
            (width, height) = hr.x_max - hr.x_min

            nc = plt.Rectangle(hr.x_min, width, height, color='b', clip_on=True, fill=False)
            plt.plot(hr.x_mid[0], hr.x_mid[1], 'x', markersize=5, color=marker[idx][0], markeredgewidth=2)

            fig.gca().add_artist(nc)

        plt.title(title)
        plt.axis(plot_instance.axis)

        return plt
