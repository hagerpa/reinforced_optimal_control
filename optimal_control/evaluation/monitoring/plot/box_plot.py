import numpy as np
from matplotlib import pyplot as plt


class BoxPlot:
    def __init__(self, df, main_axis_keys, main_axis_values,
                 box_width=0.2, space_between=0.2):

        self.box_width = box_width
        self.space_between = space_between
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('lower bound')
        ax2 = plt.twinx()
        ax2.set_ylabel('cpu time')
        ax1.figure.set_figwidth(15)
        ax1.figure.set_figheight(10)
        ax1.set_xticklabels(main_axis_values)
        ax1.set_xlabel(main_axis_keys)

        self.COLORS = ['black', 'b', 'r', 'g', 'purple', 'orange', 'violet', 'cyan']
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.df = df
        self.main_axis_values = main_axis_values
        self.main_axis_keys = main_axis_keys

    def plot_res(self, n, pos, **instance):
        self.ax1.set_xticks(np.arange(len(self.main_axis_values)) * (n * 0.2 + self.space_between))

        for (i, val) in enumerate(self.main_axis_values):
            sub_query = ' & '.join([('{0} == "{1}"' if type(v) == str else '{0} == {1}').format(k, v) for k, v in
                                    zip(self.main_axis_keys, val)])
            if len(instance.items()) > 0:
                sub_query += ' & ' + ' & '.join(
                    [('{0} == "{1}"' if type(v) == str else '{0} == {1}').format(k, v) for k, v in
                     instance.items()])

            sdf = self.df.query(sub_query)
            if len(sdf) == 0:
                continue
            else:
                item = sdf.iloc[0]
                err = item['te_tr_err'] + item['te_te_err']

                pos_ = self.ax1.get_xticks()[i] + pos * self.box_width

                self.ax2.bar(pos_, item['cpu_time'], width=0.1, color='grey', alpha=0.3)
                rectangle = plt.Rectangle((pos_ - 0.1, item['te_mean'] - 3 * err), 0.2, 2 * 3 * err,
                                          color=self.COLORS[pos], alpha=0.3)
                self.ax1.add_patch(rectangle)
                attrs = {"color": self.COLORS[pos], "linewidth": 1.0}
                if i == 0: attrs.update(label=instance)
                self.ax1.hlines(item['te_mean'], pos_ - 0.1, pos_ + 0.1, **attrs)
