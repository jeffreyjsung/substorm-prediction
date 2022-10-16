import itertools
import json
import os
import pickle
from datetime import datetime, timedelta

import dataHandler
import requests
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl
import torch
from torchvision import transforms
from common import open_figure, save_figure, moving_average, determine_grid
from console_progressbar import ProgressBar
from dataHandler.datasets import AsimDataSet
from dataHandler.logger import logger
from dataHandler.DataHandler import DataHandler
from matplotlib import patches, gridspec
from matplotlib.patches import Ellipse
from sklearn.metrics import mean_absolute_error as mae, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split as ttsplit
import scipy.interpolate
from scipy.signal import savgol_filter

from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import KernelDensity
import collections
import umap
import joblib


class DataAnalyzer(DataHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimal_inds = [625, 970, 314, 859, 697, 596, 807, 735, 775, 226, 294, 678, 254, 878,
                             706, 319, 309, 212, 142, 356, 470, 713, 364, 381, 631, 645, 663, 518,
                             595, 949]
        self.pp = dataHandler.PreProcessor(ceil_path=self.ceil_path, magn_path=self.magn_path,
                                           asim_path=self.asim_path, oath_path=self.oath_path)

        self.fp = dataHandler.Provider(ceil_path=self.ceil_path, magn_path=self.magn_path,
                                       asim_path=self.asim_path, oath_path=self.oath_path)
    """
    Functions that read in data and preprocess them.
    """
    def preprocess_asim_feat_data(self) -> pd.DataFrame():
        # files were preprocessed and combined on polar, then rsynced over
        date_start = datetime(year=2010, month=11, day=1)
        date_end = datetime(year=2011, month=2, day=28)
        classes = list(self.classes_6.keys())
        if os.path.exists(self.data_path + "asim_feats_10-11_predicted.hdf"):
            asim_magn = pd.DataFrame(pd.read_hdf(self.data_path +
                                                  "asim_feats_10-11_predicted.hdf"))
            logger.info("\nlength of dataset: {} \noccurence of classes:\n{}".format(len(
                asim_magn), asim_magn["pred"].value_counts().sort_index()))
            return asim_magn
        asim_data = self.fp.get_asim(date_start=date_start, date_end=date_end, location="NYA")
        cloud_data = self.fp.get_ceil(date_start=date_start, date_end=date_end, location="NYA")
        magn_data = self.fp.get_magn(date_start=date_start, date_end=date_end, location="NAL")
        asim_clouds = self.fp.combine_data_sets(asim_data, cloud_data, "CBH")
        asim_magn = self.fp.combine_data_sets(asim_clouds, magn_data, ["X", "Y", "Z"], 60)
        del asim_clouds
        del asim_data
        del cloud_data
        del magn_data
        feats = asim_magn.loc[:, self.feat_names]
        preds = self.pp.predict_image_proba(feats)
        pred_class = np.argmax(preds, axis=1)
        asim_magn.loc[:, classes] = preds
        asim_magn.loc[:, "pred"] = pred_class

        asim_magn["seconds"] = asim_magn.apply(lambda row: self.row_times_to_seconds(row), axis=1)
        asim_magn.to_hdf(self.data_path + "/asim_feats_10-11_predicted.hdf", "asim_magn", "w",
                         format="fixed")
        return asim_magn

    def preprocess_correllated_data(self):
        if os.path.exists(self.data_path + "/asim_feat_corr.hdf"):
            corrs = pd.read_hdf(self.data_path + "/asim_feat_corr.hdf")
            return corrs
        classes = list(self.classes_6.keys())
        pred = "pred"
        data = self.preprocess_asim_feat_data().copy()
        to_corr = data.loc[:, classes + [pred] + self.feat_names].copy()
        corrs = to_corr.corr()
        corrs.to_hdf(self.data_path + "/asim_feat_corr.hdf", "asim_corr", "w", format="fixed")
        return corrs

    def get_all_magn_data(self):
        if os.path.exists(self.data_path + "/magn_all_comb.hdf"):
            return pd.read_hdf(self.data_path + "/magn_all_comb.hdf")
        date_start = datetime(year=1993, month=10, day=1)
        date_end = datetime(year=2020, month=11, day=30)
        magn_data = self.fp.get_magn(date_start=date_start, date_end=date_end, location="NAL")
        magn_data.to_hdf(self.data_path + "/magn_all_comb.hdf", "magn_comb", "w", format="fixed")
        return magn_data

    def __preprocess_asim_cloud_meta_data(self):
        if os.path.exists(self.data_path + "asim_ceil_metadata.hdf"):
            return pd.DataFrame(pd.read_hdf(self.data_path + "asim_ceil_metadata.hdf"))
        location = "NYA"
        date_start = datetime(year=1992, month=9, day=1)
        date_end = datetime(year=2017, month=7, day=31)
        asim_links = self.fp.asim_links_file().copy().reset_index(drop=True)
        ceil_data = self.fp.get_ceil(date_start=date_start, date_end=date_end, location=location)
        asim_ceil = self.fp.combine_data_sets(asim_links, ceil_data, "CBH")
        asim_ceil.to_hdf(self.data_path + "asim_ceil_metadata.hdf", "asim_ceil", "w",
                         format="fixed")
        return asim_ceil

    def __daily_ceil_height_info(self):
        if os.path.exists(self.data_path + "daily_height_info.hdf"):
            return pd.DataFrame(pd.read_hdf(self.data_path + "daily_height_info.hdf"))
        location = "NYA"
        date_start = datetime(year=1992, month=9, day=1)
        date_end = datetime(year=2017, month=7, day=31)
        all_ceil = self.fp.get_ceil(date_start=date_start, date_end=date_end, location=location)
        day_info = pd.DataFrame(columns=["YYYY", "MM", "DD", "AVG", "NC", "REL"])
        days = self.date_iterator(date_start, date_end, timedelta(days=1))
        pb = ProgressBar(total=sum(1 for _ in days), prefix='', suffix='', decimals=3,
                         length=50, fill='=', zfill='>')
        for date in self.date_iterator(date_start, date_end, timedelta(days=1)):
            year = date.year
            month = date.month
            day = date.day
            ceil_data = all_ceil.loc[(all_ceil["YYYY"] == year) & (all_ceil["MM"] == month) &
                                     (all_ceil["DD"] == day), :]
            if ceil_data.empty:
                day_info = day_info.append({"YYYY": date.year, "MM": date.month, "DD": date.day,
                                            "AVG": -1, "NC": 0, "REL": 0}, ignore_index=True)
                continue
            avg = np.mean(ceil_data[np.array(ceil_data["CBH"] != 32767)
                                    & np.array(ceil_data["CBH"] != -1)].CBH)
            count = np.sum(ceil_data["CBH"] == 32767)
            day_info = day_info.append({"YYYY": date.year, "MM": date.month, "DD": date.day,
                                        "AVG": avg, "NC": count, "REL": count / len(ceil_data)},
                                       ignore_index=True)
            pb.next()
        day_info = day_info.astype({"YYYY": np.int16, "MM": np.int8, "DD": np.int8, "AVG": float,
                                    "NC": np.int16, "REL": float})
        day_info.to_hdf(self.data_path + "daily_height_info.hdf", "dhinf", "w", format="fixed")
        return day_info

    """
    Helper functions
    """

    def find_best_features(self, data, column, tot_feat):
        clf = skl.linear_model.Ridge(alpha=10)
        y_data = np.array(data.loc[:, column])
        y_data = y_data - np.mean(y_data)
        idxs = []
        for j in range(tot_feat):
            print(j)
            min_mae = np.inf
            best_ind = 0
            for i in range(len(self.feat_names)):
                if i in idxs:
                    continue
                feats = idxs.copy()
                feats.append(i)
                x_data = np.array(data.loc[:, [self.feat_names[l] for l in feats]])
                x_train, x_test, y_train, y_test = ttsplit(x_data, y_data, test_size=.2,
                                                           random_state=42)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                maer = mae(y_test, y_pred)
                if maer < min_mae:
                    best_ind = i
                    min_mae = maer
            idxs.append(best_ind)
        print(min_mae)
        print(idxs)
        improved = False
        while not improved:
            x_data = np.array(data.loc[:, [self.feat_names[l] for l in idxs]])
            x_train, x_test, y_train, y_test = ttsplit(x_data, y_data, test_size=.2, random_state=42)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            maer = mae(y_test, y_pred)
            improved = True
            for j in range(len(idxs)):
                print(j)
                for i in range(len(self.feat_names)):
                    feats = idxs.copy()
                    feats[j] = i
                    x_data = np.array(data.loc[:, [self.feat_names[l] for l in feats]])
                    x_train, x_test, y_train, y_test = ttsplit(x_data, y_data, test_size=.2,
                                                               random_state=42)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    curr_mae = mae(y_test, y_pred)
                    if curr_mae < maer:
                        print(curr_mae)
                        maer = curr_mae
                        idxs[j] = i
                        improved = False
            print(improved)
            print(idxs)
            print(maer)
        return idxs


    """
    Plotting functions
    """

    def plot_part(self, i_start, i_end, sorted_inds, y_data, y_test, y_pred, n, times, column,
                  name="", loc="lower left"):
        fontsize = 15
        label_font = {'size': str(fontsize)}
        secs = mdate.epoch2num(times)
        date_fmt = '%y-%m-%d %H:%M'
        date_formatter = mdate.DateFormatter(date_fmt)
        dev = np.abs(y_pred - y_test)
        avg_pred = savgol_filter(y_pred, n, 5)
        avg_dev = np.abs(avg_pred - y_test)
        inds = np.arange(i_start, i_end, 1)
        mse_all = mse(y_test[inds], y_pred[inds])
        mae_all = mae(y_test[inds], y_pred[inds])
        r2_all = r2(y_test[inds], y_pred[inds])
        mse_avg = mse(y_test[inds], avg_pred[inds])
        mae_avg = mae(y_test[inds], avg_pred[inds])
        r2_avg = r2(y_test[inds], avg_pred[inds])
        mean = np.mean(y_test[inds])
        print("Errors for prediction between indices {}-{}. mse:{:.2f}, mae: {:.2f}, r^2: {:.2f}"
              .format(i_start, i_end, mse_all, mae_all, r2_all))
        print("Errors for filtered prediction between indices {}-{}. mse:{:.2f}, mae: {:.2f},"
              "r^2: {:.2f}".format(i_start, i_end, mse_avg, mae_avg, r2_avg))
        print("Standard deviation in test data: {:.2f}".format(np.std(y_test[inds])))
        f1, ax = open_figure()
        """ax.plot_date(secs[data_ind], y_data[data_ind], linewidth=1, linestyle='solid', markersize=0,
                     label="all data")"""
        ax.plot_date(secs[sorted_inds[inds]], y_test[inds], ".", markersize=5, label="test data")
        ax.plot_date(secs[sorted_inds[inds]], y_pred[inds], '.', markersize=5, label="prediction")
        if not name.__contains__("bin"):
            ax.plot_date(secs[sorted_inds[inds]], avg_pred[inds], linewidth=1,
                         linestyle='solid', markersize=0, label="filtered prediction")
            ax.plot_date(secs[sorted_inds[inds]], avg_dev[inds], linewidth=1,
                         linestyle='solid', markersize=0, label="absolute error")
            ax.set_ylabel(r'$B_{}$ [nT]'.format(column), **{'size': '15'})
        else:
            ax.plot_date(secs[sorted_inds[inds]], dev[inds], linewidth=1,
                         linestyle='solid', markersize=0, label="absolute deviation")
            ax.set_ylabel(r'$B_{}$ [bin number]'.format(column), **{'size': '15'})
        ax.tick_params(labelsize=fontsize)
        y_lims = ax.get_ylim()
        y_len = np.diff(y_lims)
        ax.set_ylim(y_lims[0] - .4 * y_len, y_lims[1])
        ax.legend(loc=loc, fontsize=fontsize)
        ax.grid()
        ax.xaxis.set_major_formatter(date_formatter)
        f1.autofmt_xdate()
        f1.tight_layout()
        f1.show()
        if name != "":
            save_figure(f1, name+"_preds", (50, 50))
        plt.close(f1)

        bins = np.linspace(np.min(y_data), np.max(y_data), 1000)

        kde_test = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_pred = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_avg = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_all = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_test.fit(y_test[inds].reshape((-1, 1)))
        kde_pred.fit(y_pred[inds].reshape((-1, 1)))
        kde_avg.fit(avg_pred[inds].reshape((-1, 1)))
        kde_all.fit(np.array(y_data).reshape((-1, 1)))
        log_dens_test = kde_test.score_samples(bins.reshape(-1, 1))
        log_dens_pred = kde_pred.score_samples(bins.reshape(-1, 1))
        log_dens_avg = kde_avg.score_samples(bins.reshape(-1, 1))
        log_dens_all = kde_all.score_samples(bins.reshape(-1, 1))

        ymax = np.exp(np.max([log_dens_test, log_dens_pred, log_dens_avg, log_dens_all]))

        left, width = 0.125, 0.60
        bottom, height = 0.1, 0.60
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.24]
        rect_histy = [left + width + spacing, bottom, 0.24, height]
        rect_comb_hist = [left + width + spacing, bottom + height + spacing, .24, .24]

        # start with a square Figure
        f2, ax = open_figure()
        ax.remove()
        ax = f2.add_axes(rect_scatter)
        ax_histx = f2.add_axes(rect_histx)
        ax_histy = f2.add_axes(rect_histy)
        ax_hist_comb = f2.add_axes(rect_comb_hist)
        s1 = ax.scatter(y_pred[inds][0], y_test[inds][0], label="test data", color="C2")
        s2 = ax.scatter(y_pred[inds][0], y_test[inds][0], label="all data", color="C3")
        s3 = ax.scatter(y_pred[inds], y_test[inds], label="single predictions", color="C0")
        s4 = ax.scatter(avg_pred[inds], y_test[inds], label="filtered predictions",
                        color="C1")
        ax.set_xlabel(r"$B_{}$ [nT] - predicted values".format(column), **{'size': '15'})
        ax.set_ylabel(r"$B_{}$ [nT] - test values".format(column), **{'size': '15'})
        ax.grid()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([np.min([xlim, ylim]), np.max([xlim, ylim])],
                [np.min([xlim, ylim]), np.max([xlim, ylim])], color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax_histx.plot(bins, np.exp(log_dens_pred))
        ax_histx.plot(bins, np.exp(log_dens_avg))
        ax_histx.grid()
        ax_histx.set_xlim(xlim)
        ax_histx.set_xticklabels([])
        ax_histx.set_ylim([-0.001, ymax + 0.003])
        ax_histx.set_ylabel("Normalized Density", **{'size': '15'})
        ax_histy.plot(np.exp(log_dens_test), bins, color="C2")
        ax_histy.grid()
        ax_histy.set_ylim(ylim)
        ax_histy.set_yticklabels([])
        ax_histy.set_xlim([-0.001, ymax + 0.003])
        brange = [np.argmin(np.abs(bins-np.min([xlim, ylim]))),
                  np.argmin(np.abs(bins-np.max([xlim, ylim])))]
        ax_hist_comb.plot(bins[brange[0]:brange[1]], np.exp(log_dens_pred[brange[0]:brange[1]]))
        ax_hist_comb.plot(bins[brange[0]:brange[1]], np.exp(log_dens_avg[brange[0]:brange[1]]))
        ax_hist_comb.plot(bins[brange[0]:brange[1]], np.exp(log_dens_test[brange[0]:brange[1]]))
        ax_hist_comb.plot(bins[brange[0]:brange[1]], np.exp(log_dens_all[brange[0]:brange[1]]))
        ax_hist_comb.grid(axis="y")
        ax_hist_comb.set_yticks(ax_histy.get_xticks())
        ax_hist_comb.set_yticklabels([])
        ax_hist_comb.set_ylim([-0.001, ymax + 0.003])
        lns = [s1, s2, s3, s4]
        labs = [l.get_label() for l in lns]
        ax_histx.legend(lns, labs, loc="upper left", fontsize=fontsize)
        ax.tick_params(labelsize=15)
        ax_histy.tick_params(labelsize=15)
        ax_histx.tick_params(labelsize=15)
        f2.show()
        if name != "":
            save_figure(f2, name+"_scatter", (50, 50))
        plt.close(f2)

        return f1, f2

    def get_magn_train_test_split(self, y_data):
        if os.path.exists(self.magn_split):
            best = np.loadtxt(self.magn_split).astype("int")
            return best
        ndata = len(y_data)
        count, edges = np.histogram(y_data, range=(np.min(y_data), np.max(y_data)), bins=51)
        dist = count / ndata
        d = np.log(dist)
        d[np.isinf(d)] = 0
        id = np.arange(0, ndata//10*7, 1)
        diff = ndata
        best = id
        for i in range(ndata//100):
            id = np.remainder(id + 100, ndata)
            ldist = np.histogram(y_data[id], range=(np.min(y_data), np.max(y_data)), bins=51)[0]/len(id)
            ldist = np.log(ldist)
            ldist[np.isinf(ldist)] = 0
            ldiff = np.sum((ldist - d)**2)
            if ldiff < diff:
                diff = ldiff
                best = id
        np.savetxt(self.magn_split, best, delimiter=",", fmt='%s')
        return best

    def print_magn_train_test_split(self, y_data, y_train, y_test, column):
        bins = np.linspace(np.min(y_data), np.max(y_data), 1000)
        kde_test = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_train = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_all = KernelDensity(kernel='gaussian', bandwidth=10)
        kde_test.fit(y_test.reshape((-1, 1)))
        kde_train.fit(np.array(y_train).reshape((-1, 1)))
        kde_all.fit(np.array(y_data).reshape((-1, 1)))
        log_dens_test = kde_test.score_samples(bins.reshape(-1, 1))
        log_dens_train = kde_train.score_samples(bins.reshape(-1, 1))
        log_dens_all = kde_all.score_samples(bins.reshape(-1, 1))

        fig, ax = open_figure(constrained_layout=True)
        ax.plot(bins, np.exp(log_dens_all)+1e-8, label="all data")
        ax.plot(bins, np.exp(log_dens_train)+1e-8, label="training data")
        ax.plot(bins, np.exp(log_dens_test)+1e-8, label="testing data")
        ax.legend()
        ax.set_xlabel(r"$B_{}$ [nT]".format(column))
        ax.set_ylabel("Normalized Density")
        ax.set_yscale("log")
        fig.show()
        save_figure(fig, self.image_path + "magn_train_dist")

    def classify_and_print(self, data, feats, column: str, part, degree: int = 1, n_bin=0):
        x_data = data.loc[:, feats]
        if column == "H":
            X = data.loc[:, "X"]
            Y = data.loc[:, "Y"]
            y_data = np.sqrt(X**2+Y**2)
            y_data = y_data.reset_index(drop=True)
        else:
            y_data = data.loc[:, column].reset_index(drop=True)
        mean = np.mean(y_data)
        y_data = y_data - mean

        poly = skl.preprocessing.PolynomialFeatures(degree=degree)
        poly_data = skl.preprocessing.scale(poly.fit_transform(x_data), axis=1)

        best = self.get_magn_train_test_split(y_data)

        inds = y_data.index
        x_train = poly_data[best]
        x_test = np.delete(poly_data, best, axis=0)
        y_train = y_data[best]
        y_test = np.delete(np.array(y_data), best)
        test_inds = np.delete(np.array(inds), best)
        if not os.path.exists(self.image_path + "magn_train_dist.png"):
            self.print_magn_train_test_split(y_data, y_train, y_test, column)

        sort_index = np.argsort(test_inds)
        sorted_inds = test_inds[sort_index]
        x_test = x_test[sort_index]
        y_test = np.array(y_test)[sort_index]
        times = np.array(data.loc[:, "seconds"])

        clf = skl.linear_model.Ridge(alpha=1, normalize=False, random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        n = 51
        # all of strong event at end
        f1, f2 = self.plot_part(45200, 47230, sorted_inds, y_data, y_test, y_pred, n, times, column,
                                self.image_path + "magn/peaks_{}_deg_{}_{}".
                                format(part, degree, column))
        f3, f4 = self.plot_part(n // 2, len(x_test) - n // 2, sorted_inds, y_data, y_test, y_pred,
                                n, times, column, self.image_path + "magn/all_{}_deg_{}_{}".
                                format(part, degree, column))

        if n_bin == 0:
            return f1, f2, f3, f4

        y_hist_test = - y_test + np.max(y_data)
        y_hist_train = - np.array(y_train) + np.max(y_data)
        m = np.max(np.append(y_hist_train, y_hist_test))
        y_hist = - np.array(y_data) + np.max(y_data)
        bins = np.logspace(1, np.log10(m), n_bin)
        hist_test = np.argmin(np.abs(y_hist_test - np.expand_dims(bins, 1)), axis=0)
        hist_train = np.argmin(np.abs(y_hist_train - np.expand_dims(bins, 1)), axis=0)
        hist = np.argmin(np.abs(y_hist - np.expand_dims(bins, 1)), axis=0)
        clf = skl.linear_model.RidgeClassifier(alpha=20)
        clf.fit(x_train, hist_train)
        hist_pred = clf.predict(x_test)
        pred_hist = sorted(collections.Counter(np.round(hist_pred).astype(int)).items())
        (pred_bins, pred_prob) = zip(*pred_hist)
        test_hist = sorted(collections.Counter(hist_test).items())
        (test_bins, test_prob) = zip(*test_hist)
        n = 10
        f5, f6 = self.plot_part(45200, 46758, sorted_inds, hist, hist_test, hist_pred, n, times,
                                column, self.image_path+"magn/peaks_{}_deg_{}_{}_nbins_{}".
                                format(part, degree, column, n_bin))
        f7, f8 = self.plot_part(n // 2, len(x_test) - n//2, sorted_inds, hist, hist_test, hist_pred,
                                n, times, column, self.image_path+"magn/all_{}_deg_{}_{}_nbins_{}".
                                format(part, degree, column, n_bin))

        f9, ax = open_figure()
        ax.plot(bins[np.array(pred_bins)], pred_prob/np.sum(pred_prob))
        ax.plot(bins[np.array(test_bins)], test_prob/np.sum(test_prob))
        ax.set_xscale("log")
        ax.set_xlabel("bin values")
        ax.set_ylabel("bin probability")
        ax.legend(["prediction", "test"])
        ax.set_title("magn/binned_prediction_{}_deg_{}_{}_nbins_{}".
                     format(part, degree, column, n_bin))
        f9.show()
        f9.savefig(self.image_path+"magn/binned_prediction_{}_deg_{}_{}_nbins_{}.svg".
                   format(part, degree, column, n_bin))
        return f1, f2, f3, f4, f5, f6, f7, f8, f9

    def plot_magn_predictions(self):
        asim_magn = self.preprocess_asim_feat_data()
        clear_skies = asim_magn.loc[:, "CBH"] >= 2000
        with_nl = asim_magn.loc[:, "pred"] <= 2
        data = asim_magn.loc[with_nl & clear_skies, :].copy().\
            sort_values(["YYYY", "MM", "DD", "SS"])
        del asim_magn, clear_skies, with_nl

        reduced_feat_names = [self.feat_names[i] for i in self.optimal_inds]
        fs = pd.DataFrame(columns=["X", "Y", "Z", "H"], index=np.arange(0, 10))
        for c in ["X", "Y", "Z"]:
            figs = []
            figs.append(self.classify_and_print(data, self.feat_names, c, "all"))
            figs.append(self.classify_and_print(data, reduced_feat_names, c, "reduced"))
            figs.append(self.classify_and_print(data, reduced_feat_names, c, "reduced", 2))
            figs.append(self.classify_and_print(data, reduced_feat_names, c, "reduced", 3))
            figs.append(self.classify_and_print(data, reduced_feat_names[:10], c, "reduced_10", 5))
            figs.append(self.classify_and_print(data, self.feat_names, c, "all", 1, 2))
            figs.append(self.classify_and_print(data, self.feat_names, c, "all", 1, 3))
            figs.append(self.classify_and_print(data, self.feat_names, c, "all", 1, 10))
            figs.append(self.classify_and_print(data, self.feat_names, c, "all", 1, 20))
            figs.append(self.classify_and_print(data, self.feat_names, c, "all", 1, 50))
            fs[c] = figs
        return fs

    def plot_daily_cloud_coverage(self):
        day_info = self.__daily_ceil_height_info()
        day_info = day_info[(day_info["MM"] == 11) | (day_info["MM"] == 12) | (day_info["MM"] == 1)
                            | (day_info["MM"] == 2)]
        dates = day_info["YYYY"].astype(str) + "-" + day_info["MM"].astype(str) + "-" + \
                day_info["DD"].astype(str)
        x_data = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
        yeardates = []
        years = []
        avgrels = []
        avgheights = []
        shists = []
        bins = np.linspace(start=0, stop=100, num=11) / 100
        for date in self.date_iterator(datetime(year=1992, month=12, day=31),
                                       datetime(year=2016, month=12, day=31), timedelta(days=365)):
            date = date.date()
            min_date = date - timedelta(days=70)
            max_date = date + timedelta(days=70)
            yeardates.append(date)
            years.append(date.year)
            season_data = day_info[(np.array(x_data) >= min_date) & (np.array(x_data) <= max_date)]
            avgrel = np.mean(season_data.REL)
            avgheight = np.mean(season_data.AVG)
            s_hist = np.histogram(season_data["REL"], bins)
            avgrels.append(avgrel)
            avgheights.append(avgheight)
            shists.append(s_hist[0] / np.sum(s_hist[0]))
        s_hists = np.array(shists).T[::-1]
        labels = [None if np.remainder(i * 100, 10) else int(i * 100) for i in bins[::-1]]
        year_ticks = [year - np.min(years) + 0.5 for year in years]

        fig, ax1 = open_figure()
        plt.tick_params(bottom='on')
        cbar_ax = fig.add_axes([.88, .08, .05, .9])
        fig.subplots_adjust(right=.8, bottom=.08, top=.98)
        ax1 = sns.heatmap(s_hists, cmap=sns.color_palette("viridis", as_cmap=True),
                          yticklabels=labels, xticklabels=years, cbar_ax=cbar_ax, ax=ax1,
                          cbar_kws={"label": "percentage"})
        ax2 = ax1.twinx()
        sns.lineplot(x=year_ticks, y=[-(avg * 10 - 10) for avg in avgrels], linewidth=5, ax=ax1,
                     color="b")
        sns.lineplot(x=year_ticks, y=avgheights, linewidth=5, ax=ax2, color="r")
        ax1.axis('tight')
        ax1.set_ylabel("Amount of 'clear skies' per day (heatmap) and seasonal average (blue)")
        ax2.set_ylabel("Average seasonal cloud base height (CBH) [m] (red)")
        ax1.set_xlabel("Season - first year displayed - only nov-feb data used")
        fig.show()
        fig.savefig(self.image_path+"ceil/cloud_coverage.svg")

    def plot_asim_cloud_coverage(self):
        cloud_data = self.__preprocess_asim_cloud_meta_data()
        bins = np.append(np.linspace(start=0, stop=30000, num=31), 32767)
        hist = np.histogram(cloud_data["CBH"], bins=bins)

        fig, ax = open_figure()
        ax.bar(hist[1][1:] - 500, hist[0] / np.sum(hist[0]), width=900)
        ax.set_xlabel("CBH [m]")
        ax.set_ylabel("Occurence")
        plt.show()
        fig.savefig(self.image_path+"ceil/CBH_hist.svg")

        date_start = datetime(year=2006, month=1, day=1)
        date_end = datetime(year=2017, month=3, day=1)

        days = []
        cloud_free_time = []
        high_cloud_time = []
        for date in self.date_iterator(date_start, date_end, timedelta(days=1)):
            days.append(date.date())
            year = date.year
            month = date.month
            day = date.day
            data = cloud_data.loc[(cloud_data["YYYY"] == year) & (cloud_data["MM"] == month) &
                                  (cloud_data["DD"] == day)].copy()
            if data.empty:
                cloud_free_time.append(0)
                high_cloud_time.append(0)
                continue
            tc = data.shape[0]
            cloud_free = np.sum(data["CBH"] == 32767)
            high_cloud = np.sum(data["CBH"] >= 2000)
            cloud_free_time.append(cloud_free / tc)
            high_cloud_time.append(high_cloud / tc)

        n = 31
        avgs_free = moving_average(cloud_free_time, n)
        avgs_high = moving_average(high_cloud_time, n)

        def plot(y_data_1, y_data_2, y_label, name):
            fig, ax1 = open_figure()
            ax2 = ax1.twinx()
            ax1.plot(days, y_data_1, color="b")
            ax2.plot(days[int(np.floor(n / 2)):-int(np.floor(n / 2))], y_data_2, color="r")
            ax1.set_xlabel('year')
            ax1.set_ylabel(y_label)
            ax2.set_ylabel(str(n) + ' day rolling average')
            fig.tight_layout()
            plt.show()
            fig.savefig(self.image_path+"ceil/"+name+".svg")

        plot(cloud_free_time, avgs_free, "Daily average percentage of no cloud coverage",
             "cloud_coverage_asim")
        plot(high_cloud_time, avgs_high, "Daily average percentage of high cloud (>=5km) coverage",
             "high_clouds")

    def plot_asim_availability(self):
        ind = self.fp.asim_availability_file()
        ind["DATE"] = ind.apply(lambda row: datetime(year=int(row["YYYY"]), month=int(row["MM"]),
                                                     day=int(row["DD"])).date(), axis=1)
        dates = []
        y_values = []
        a = np.log(-1)
        for date in self.date_iterator(datetime(year=2006, month=1, day=1),
                                       datetime(year=2020, month=10, day=17), timedelta(days=1)):
            date = date.date()
            dates.append(date)
            if np.sum(ind["DATE"] == date) >= 1:
                y_values.append(1)
            else:
                y_values.append(a)

        bardata = pd.DataFrame()
        bardata["DATES"] = dates
        bardata["values"] = y_values
        bardata.set_index("DATES")

        fig, ax = open_figure()
        ax.plot(dates, y_values, linewidth=1)
        ax.set_xlabel('date')
        ax.set_ylabel('Availability of ASIM image date (true / false)')
        fig.tight_layout()
        fig.show()
        fig.savefig(self.image_path+"/asim/asim_avil.svg")

    def plot_cloud_predictions_roc(self):
        asim_magn = self.preprocess_asim_feat_data()
        fontsize = 15
        label_font = {'size': str(fontsize)}
        preds = np.array(asim_magn.loc[:, "class_3"])
        cbh_thr = 2000
        proba = np.linspace(0, 1, 1001)
        count = np.zeros(proba.shape)
        for i, prob in enumerate(proba):
            count[i] = np.sum((preds <= prob) == (asim_magn.loc[:, "CBH"] >= cbh_thr))
        fpr, tpr, thresholds = roc_curve(asim_magn.loc[:, "CBH"] >= cbh_thr, 1 - preds)
        tpr95l = np.argmin(np.abs(tpr-.95))
        fpr05l = np.argmin(np.abs(fpr-.05))
        acc95l = np.argmin(np.abs(proba-thresholds[tpr95l]))
        acc05l = np.argmin(np.abs(proba-thresholds[fpr05l]))
        l = np.argmax(count / len(asim_magn))
        logger.info("\n"+classification_report(asim_magn.loc[:, "CBH"] <= cbh_thr,
                                          asim_magn.loc[:, "pred"] == 3, digits=4))
        logger.info("Area under ROC curve: {:.4f}".format(np.trapz(tpr, fpr)))
        logger.info("Maximum accuracy achieved at {:.2f}% with an accuracy of {:.2f}%".format(
            proba[l]*100, count[l]*100/len(asim_magn)))

        fig, ax = open_figure(constrained_layout=True)
        ax.plot(proba, count / len(asim_magn))
        ax.plot([0, 1], [np.sum((asim_magn.loc[:, "CBH"] <= cbh_thr)) / len(asim_magn),
                         np.sum((asim_magn.loc[:, "CBH"] >= cbh_thr)) / len(asim_magn)], '--')
        ax.plot(proba[acc95l], count[acc95l] / len(asim_magn), 'x', markersize=10)
        ax.text(proba[acc95l] + .07, count[acc95l] / len(asim_magn) - .07,
                str("({:.2f}, {:.2f})".format(proba[acc95l], count[acc95l] / len(asim_magn))),
                **label_font)
        ax.plot(proba[acc05l], count[acc05l] / len(asim_magn), 'x', markersize=10)
        ax.text(proba[acc05l] - .3, count[acc05l] / len(asim_magn) - .06,
                str("({:.2f}, {:.2f})".format(proba[acc05l], count[acc05l] / len(asim_magn))),
                **label_font)
        ax.set_xlabel("{} probability threshold".format(self.classes_6.get("class_3").get("name")),
                      **label_font)
        ax.set_ylabel("accuracy".format(cbh_thr), **label_font)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend(["thresholded prediction", "random prediction"])
        fig.show()
        save_figure(fig, self.image_path+"asim/acc_cloud_data")

        fig, ax = open_figure(constrained_layout=True)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], '--')
        ax.plot(fpr[tpr95l], tpr[tpr95l], 'x', markersize=10)
        ax.text(fpr[tpr95l]+.05, tpr[tpr95l]-.02,
                str("({:.2f}, {:.2f})".format(fpr[tpr95l], tpr[tpr95l])), **label_font)
        ax.plot(fpr[fpr05l], tpr[fpr05l], 'x', markersize=10)
        ax.text(fpr[fpr05l] + .05, tpr[fpr05l] - .02,
                str("({:.2f}, {:.2f})".format(fpr[fpr05l], tpr[fpr05l])), **label_font)
        ax.set_xlabel("false positive rate", **label_font)
        ax.set_ylabel("true positive rate", **label_font)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        fig.show()
        save_figure(fig, self.image_path+"asim/roc_cloud_data")

        fig, ax = open_figure(constrained_layout=True)
        ax.plot(thresholds, tpr)
        ax.plot(thresholds, fpr)
        plt.xlim([0, 1])
        ax.set_xlabel("threshold", **label_font)
        ax.set_ylabel("probabilities", **label_font)
        ax.grid()
        ax.legend(["true positive rate", "false positive rate"])
        fig.show()
        save_figure(fig, self.image_path+"asim/tprate_vs_threshold", (40, 40))

    def get_cluster_files(self):
        umap_all_path = self.data_path + "reductions/umap_embed.hdf"
        umap_reducer_path = self.data_path + "reductions/umap_reducer"
        np.random.seed(42)

        if os.path.exists(umap_reducer_path):
            umap_reducer = joblib.load(umap_reducer_path)
            fitted = True
        else:
            umap_reducer = umap.UMAP(n_neighbors=100, learning_rate=1, n_epochs=1000,
                                     verbose=True, low_memory=False, random_state=42)
            fitted = False
        if os.path.exists(umap_all_path):
            data = pd.read_hdf(umap_all_path)
        else:
            umap_data = self.preprocess_asim_feat_data().copy().reset_index(drop=True)
            data = umap_data[self.feat_names]
            if not fitted:
                umap_reducer = umap_reducer.fit(data)
            joblib.dump(umap_reducer, umap_reducer_path)
            umap_embed = umap_reducer.transform(data)
            umap_data.loc[:, ["umap_0", "umap_1"]] = umap_embed
            umap_data.to_hdf(umap_all_path, "umap", "w", format="table")
        return data, umap_reducer

    @staticmethod
    def get_file(filename, url, dataset=None):
        if not os.path.exists(filename):
            dir = os.path.dirname(filename)
            if not os.path.exists(dir):
                os.makedirs(dir)
            resp = requests.get(url)
            resp.raise_for_status()
            open(filename, 'wb').write(resp.content)
        if dataset is None:
            return Image.open(filename)
        else:
            return dataset.load_image(filename)

    def get_umap_split(self, n):
        if os.path.exists(self.data_path + "reductions/split.csv"):
            return pd.read_csv(self.data_path + "reductions/split.csv").values.ravel()
        m = 3000
        np.random.seed(42)
        sel = np.random.choice(n, size=m, replace=False)
        np.savetxt(self.data_path + "reductions/split.csv", sel, delimiter=',', fmt='%d')
        return sel

    def plot_umap_preds(self):
        data, _ = self.get_cluster_files()
        sel = self.get_umap_split(len(data))
        sel_data = data.iloc[sel].copy()
        files = list(sel_data["FILENAME"])
        files = [os.path.split(f)[1] for f in files]
        sel_data["file"] = files
        with open(self.data_path + "reductions/ims/annotations.json") as f:
            annotations = json.load(f)

        cs = {"arc": 0, "diffuse": 1, "discrete": 2, "cloudy": 3, "moon": 4, "clear sky / no aurora": 5}
        anns = {}
        for a in annotations:
            file_upload = os.path.split(a.get("image"))[1]
            filename = "_".join(file_upload.split("_cal")[:-1])+"_cal.png"
            if not a.get("choice"):
                continue
            label = a.get("choice")
            c = cs.get(label)
            anns.update({filename: c})
        files = list(anns.keys())

        for index, value in sel_data.iterrows():
            if not os.path.exists(self.data_path + "reductions/ims/" + value["file"]):
                print("a")

        annotated = sel_data.loc[sel_data["file"].isin(files)].copy()
        annotated["annotated"] = sel_data.apply(lambda row: anns.get(row["file"]), axis=1)

        pred = np.array(annotated.loc[:, "pred"])
        ann = np.array(annotated.loc[:, "annotated"]).astype(int)
        x = np.array(annotated.loc[:, "umap_0"])
        y = np.array(annotated.loc[:, "umap_1"])
        print(confusion_matrix(ann, pred))
        print(confusion_matrix(ann <= 2, pred <= 2))
        print(classification_report(ann, pred))
        print(classification_report(ann <= 2, pred <= 2))
        fontsize = 10
        fig, ax = open_figure(constrained_layout=True)
        size = {".": 1, ",": 4}
        for i in np.unique(pred):
            for j in np.unique(ann):
                class_name = "class_" + str(j)
                loc = (pred == i) & (ann == j)
                m = ","
                if (i <= 2) == (j <= 2):
                    m = "."
                ax.scatter(x[loc], y[loc], s=size.get(m), marker=m,
                           color=self.classes_6.get(class_name).get("color"),
                           label=self.classes_6.get(class_name).get("name"))
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlim(-10, 17)
        ax.tick_params(labelsize=fontsize)
        fig.show()
        save_figure(fig, self.image_path+"asim/embeds_classified", (20, 20))

    def export_umap_images(self):
        data, _ = self.get_cluster_files()
        sel = self.get_umap_split(len(data))

        sel_data = data.iloc[sel]
        dataset = AsimDataSet([], [])
        for i in range(len(sel_data)):
            print("{}/{}".format(i, len(sel_data)))
            sel = sel_data.iloc[i]
            fname = sel["FILENAME"]
            im = self.get_file(self.asim_path + fname, sel["URL"], dataset)[1][1]
            pilim = np.array(im)
            pilim = pilim - np.min(pilim)
            pilim = pilim / np.max(pilim)
            pilim = Image.fromarray(pilim*255).convert("L")
            pilim.save(self.data_path + "reductions/ims/" + os.path.split(fname)[1])

    def demonstrate_cluster(self, clouds=False):
        data, _ = self.get_cluster_files()
        date_start = datetime.fromtimestamp(int(data["seconds"].min()))
        date_end = datetime.fromtimestamp(int(data["seconds"].max()))
        dataset = AsimDataSet([], [])
        if clouds:
            loc = "NYA"
            ceil = self.fp.get_ceil(date_start=date_start, date_end=date_end, loc=loc)
            ceil["seconds"] = ceil.apply(lambda row: self.row_times_to_seconds(row), axis=1)
        n = len(data)
        labels = np.array(data["pred"])
        umap_embed = np.array(data.loc[:, ["umap_0", "umap_1"]])

        sel2 = self.get_umap_split(len(data))
        sel_labels = labels[sel2]
        sel_embed = umap_embed[sel2]
        sel_meta = data.iloc[sel2]

        nrows = 4
        ncols = 4
        poy = nrows//2-1
        pox = ncols//2-1

        fontsize = 15
        label_font = {'size': str(fontsize)}

        fig = plt.figure(figsize=(ncols*3, nrows*3))
        available = np.ones(shape=(nrows, ncols))
        outer_grid = gridspec.GridSpec(nrows, ncols, hspace=.2, wspace=.2, left=.05, right=.95,
                                       bottom=0.05, top=.95)
        embed_grid = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                      subplot_spec=outer_grid[poy:poy+2, pox:pox+2])
        available[poy:poy+2, pox:pox+2] = 0
        ax = plt.Subplot(fig, embed_grid[0])
        for i in np.unique(sel_labels):
            class_name = "class_" + str(i)
            loc = sel_labels == i
            ax.scatter(sel_embed[loc, 0], sel_embed[loc, 1], s=2,
                       color=self.classes_6.get(class_name).get("color"),
                       label=self.classes_6.get(class_name).get("name"),
                       marker=self.classes_6.get(class_name).get("marker"))
        leg = ax.legend(fontsize=15)
        for lh in leg.legendHandles:
            lh._sizes = [15.]
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlim(-10, 17)
        ax.tick_params(labelsize=fontsize)
        fig.add_subplot(ax)
        ly = np.diff(ax.get_ylim())
        lx = np.diff(ax.get_xlim())
        r = 1.5

        areas = []
        areas.append(np.array([[0, 2], [11, 13], [-1, 2]]))
        areas.append(np.array([[4, 6], [11, 13], [0, 2.5]]))
        areas.append(np.array([[2, 4], [9, 11], [-1, 1]]))
        areas.append(np.array([[4, 6], [7, 9], [4, 1]]))
        areas.append(np.array([[0, 2], [7, 9], [1.5, -1]]))
        areas.append(np.array([[10, 12], [2, 4], [-1.5, -1]]))
        areas.append(np.array([[-6, -4], [5, 7], [0, 3]]))
        areas.append(np.array([[12.5, 14.5], [-2.5, -0.5], [-1, .5]]))
        areas.append(np.array([[-6, -4], [1, 3], [-1, -.5]]))
        areas.append(np.array([[-5, -3], [-4, -2], [-2, -.5]]))
        areas.append(np.array([[-2, 0], [5, 7], [3, -1]]))
        areas.append(np.array([[-3, -1], [2, 4], [3, 0]]))

        np.random.seed(42)
        count = 4
        for area, num in zip(areas, range(1, len(areas)+1)):
            location = np.unravel_index(np.where(available.ravel())[0][0], available.shape)
            available[location] = 0
            x_lims = area[0]
            y_lims = area[1]
            if len(area) == 3:
                line = False
                offset = area[2]
                ellipse = Ellipse((x_lims[0] + offset[0], y_lims[0] + offset[1]), color="r",
                                  width=r, height=r*ly/lx, fill=False)
                ax.add_patch(ellipse)
                ax.text(x_lims[0]+offset[0], y_lims[0]+offset[1], str(num), color="black",
                        horizontalalignment='center', verticalalignment='center', **label_font)
            else:
                line = True
            rect = patches.Rectangle((x_lims[0], y_lims[0]), np.diff(x_lims)[0],
                                     np.diff(y_lims)[0], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            zoom_sel = (sel_embed[:, 0] <= x_lims[1]) &\
                       (sel_embed[:, 0] >= x_lims[0]) & \
                       (sel_embed[:, 1] <= y_lims[1]) &\
                       (sel_embed[:, 1] >= y_lims[0])
            sel_points = sel_embed[zoom_sel]
            sel_info = sel_meta.iloc[zoom_sel]
            lcount = int(np.floor(np.sqrt(np.min([count, len(sel_points)])))**2)
            if lcount < 4:
                continue
            rnd_picks = np.random.choice(len(sel_points), lcount, replace=False)
            grid = determine_grid(lcount)
            ims_grid = gridspec.GridSpecFromSubplotSpec(grid[0], grid[1],
                                                        subplot_spec=outer_grid[location],
                                                        wspace=0.15, hspace=0.25)
            title_ax = plt.subplot(ims_grid[:, :])
            circle = plt.Circle((0.5, 0.5), radius=.06, color="r", fill=False)
            title_ax.add_patch(circle)
            title_ax.text(0.5, 0.5, str(num), horizontalalignment='center',
                          verticalalignment='center', **label_font)

            title_ax.axis("off")
            fig.add_subplot(title_ax)
            for pick, i in zip(rnd_picks, range(len(rnd_picks))):
                axi = plt.Subplot(fig, ims_grid[i])
                curr_sel = sel_info.iloc[i]
                im = np.array(
                    self.get_file(self.asim_path + curr_sel["FILENAME"], curr_sel["URL"], dataset))[1][1]
                axi.imshow(im)
                axi.axes.xaxis.set_ticks([])
                axi.get_yaxis().set_visible(False)
                pred = curr_sel["pred"]
                if clouds:
                    secs = curr_sel["seconds"]
                    cloud_loc = np.argmin(np.abs(ceil["seconds"] - secs))
                    cbh = ceil.iloc[cloud_loc]["CBH"]
                    axi.set_xlabel("{} {}".format(self.classes_6.get("class_" + str(pred)).get(
                        "name"), str(cbh)), **label_font)
                else:
                    axi.set_xlabel(self.classes_6.get("class_" + str(pred)).get("name"),
                                   **label_font)
                fig.add_subplot(axi)
            if line:
                xm = (x_lims[0], y_lims[0])
                xr = (0, 0)
                con = patches.ConnectionPatch(xyA=xm, xyB=xr, coordsA="data", coordsB="data",
                                              axesA=ax, axesB=axi, color="red")
                ax.add_artist(con)
        fig.tight_layout()
        fig.show()
        save_figure(fig, self.image_path+"asim/embeds", (25*ncols, 25*nrows))

    def plot_conf_images(self, n=6):
        def plot_images(df, title):
            for f in df["FILENAME"].to_list():
                if f not in ims:
                    ims.append(f)
            colors = [self.classes_6.get(k).get("color") for k in sorted(self.classes_6.keys())]
            fig, main_ax = open_figure(size=np.max([n * 3, 800]), constrained_layout=True)
            main_ax.set_title(title, fontdict={'size': '15'})
            main_ax.axis("off")
            ax_ar = fig.subplots(grid[1], grid[0]*2)
            ax_ar = ax_ar.ravel()
            for i in range(n):
                ax = ax_ar[i*2]
                im_loc = self.asim_path + df.iloc[i]["FILENAME"]
                url = df.iloc[i]["URL"]
                im = self.get_file(im_loc, url, dataset)[1][1]
                ima = np.array(im)
                ax.imshow(ima)
                ax.axes.xaxis.set_ticks([])
                ax.get_yaxis().set_visible(False)
                probs = df.iloc[i][list(self.classes_6.keys())]
                ax = ax_ar[i*2+1]
                ax.axvline(x=1/6, linestyle="--", color="black")
                ax.barh([1/len(self.classes_6)*k for k in reversed(range(len(self.classes_6)))],
                        probs, height=.5/len(self.classes_6), color=colors)
                ax.axis("off")
                asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
                ax.set_aspect(asp)
            fig.show()
            return fig, main_ax

        def get_ims_list():
            if os.path.exists(self.data_path+"interesting_ims.csv"):
                return pd.read_csv(self.data_path+"interesting_ims.csv").values.ravel().tolist()
            else:
                return []

        def save_ims_list(ims_list):
            np.savetxt(self.data_path+"interesting_ims.csv", ims_list, delimiter=",", fmt='%s')

        dataset = AsimDataSet([], [])
        asim_magn = self.preprocess_asim_feat_data()
        data = asim_magn.copy()
        grid = determine_grid(n)
        ims = get_ims_list()

        probs = data.loc[:, list(self.classes_6.keys())]
        entropy = np.sum(probs * np.log(probs), axis=1)
        data["entropy"] = entropy

        for j in range(len(self.classes_6)):
            curr_class = "class_" + str(j)
            data = data.sort_values(curr_class)

            lp = data.iloc[0:n]
            fig, main_ax = plot_images(lp, "Lowest probability of {}".format(self.classes_6.get(
                curr_class).get("name")))
            save_figure(fig, self.image_path+"asim/low_prob_{}_{}".format(curr_class, str(n)))

            lp = data.loc[data["pred"] == j, :]
            if len(lp) > n:
                lp = lp.iloc[0:n]
                fig, main_ax = plot_images(lp, "Lowest confidence in {}".format(self.classes_6.get(
                curr_class).get("name")))
                save_figure(fig, self.image_path+"asim/low_conf_{}_{}".format(curr_class, str(n)))

            lp = data.iloc[-n:].iloc[::-1]
            fig, main_ax = plot_images(lp, "Highest confidence in {}".format(self.classes_6.get(
                curr_class).get("name")))
            save_figure(fig, self.image_path+"asim/high_conf_{}_{}".format(curr_class, str(n)))

            data = data.sort_values("entropy")
            he = data.loc[data["pred"] == j, :].iloc[:n]
            fig, main_ax = plot_images(he, "Highest entropy in {}".format(self.classes_6.get(
                curr_class).get("name")))
            save_figure(fig, self.image_path+"asim/high_ent_{}_{}".format(curr_class, str(n)))
        save_ims_list(ims)

    def print_rbf_hyper_map(self):
        def print_countour(cs, gs, vals, levels=10, name="", title=""):
            x = np.log10(np.array(cs))
            y = np.log10(np.array(gs))
            z = np.array(vals)
            X, Y = np.meshgrid(x, y)
            rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
            Z = rbf(X, Y)
            fig, ax = open_figure(300, constrained_layout=True)
            cf = ax.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=10, cmap="viridis")
            ax.plot(x, y, 'ko', ms=3)
            cbar = plt.colorbar(cf)
            cbar.set_label("accuracy")
            ax.set_xlabel("log_10(C)")
            ax.set_ylabel(r"log_10($\gamma$)")
            ax.set_title(title)
            if name:
                save_figure(fig, self.image_path + name, (30, 30))
            plt.show()
        f = open(self.rbf_accs_file, "rb")
        accs_loaded = pickle.load(f)
        f.close()

        keys = list(accs_loaded.keys())
        cs = [k[0] for k in keys]
        gs = [k[1] for k in keys]
        test6means = [accs_loaded.get(key).get("test6means") for key in keys]
        test2means = [accs_loaded.get(key).get("test2means") for key in keys]
        train6means = [accs_loaded.get(key).get("train6means") for key in keys]
        train2means = [accs_loaded.get(key).get("train2means") for key in keys]
        test6stds = [accs_loaded.get(key).get("test6stds") for key in keys]
        test2stds = [accs_loaded.get(key).get("test2stds") for key in keys]
        train6stds = [accs_loaded.get(key).get("train6stds") for key in keys]
        train2stds = [accs_loaded.get(key).get("train2stds") for key in keys]
        cutoff6 = .67
        cutoff2 = .85
        test6means = [val if val >= cutoff6 else cutoff6 for val in test6means]
        test2means = [val if val >= cutoff2 else cutoff2 for val in test2means]
        train6means = [val if val >= cutoff6 else cutoff6 for val in train6means]
        train2means = [val if val >= cutoff2 else cutoff2 for val in train2means]
        print_countour(cs, gs, test6means, name="RBFSVM_test6means", title="RBFSVM_test6means")
        print_countour(cs, gs, train6means, name="RBFSVM_train6means", title="RBFSVM_train6means")
        print_countour(cs, gs, test2means, name="RBFSVM_test2means", title="RBFSVM_test2means")
        print_countour(cs, gs, train2means, name="RBFSVM_train2means", title="RBFSVM_train2means")
        print_countour(cs, gs, test6stds, name="RBFSVM_test6stds", title="RBFSVM_test6stds")
        print_countour(cs, gs, test2stds, name="RBFSVM_test2stds", title="RBFSVM_test2stds")
        print_countour(cs, gs, train6stds, name="RBFSVM_train6stds", title="RBFSVM_train6stds")
        print_countour(cs, gs, train2stds, name="RBFSVM_train2stds", title="RBFSVM_train2stds")

    def print_linear_hyperplots(self):
        f = open(self.linear_accs_file, "rb")
        accs = pickle.load(f)
        f.close()
        test6means = []
        test2means = []
        train6means = []
        train2means = []
        test6stds = []
        test2stds = []
        train6stds = []
        train2stds = []
        Cs = [c for c in accs]
        Cs.sort()
        for c in Cs:
            acc = accs.get(c)
            test6means.append(acc.get("test6means"))
            test2means.append(acc.get("test2means"))
            train6means.append(acc.get("train6means"))
            train2means.append(acc.get("train2means"))
            test6stds.append(acc.get("test6stds"))
            test2stds.append(acc.get("test2stds"))
            train6stds.append(acc.get("train6stds"))
            train2stds.append(acc.get("train2stds"))

        test6means = np.array(test6means)
        test2means = np.array(test2means)
        train6means = np.array(train6means)
        train2means = np.array(train2means)
        test6stds = np.array(test6stds)
        test2stds = np.array(test2stds)
        train6stds = np.array(train6stds)
        train2stds = np.array(train2stds)
        Cs = np.array(Cs)

        fig, ax = open_figure(constrained_layout=True)
        ax.errorbar(Cs, test6means, yerr=test6stds, capsize=10, label="6 class test")
        ax.errorbar(Cs, test2means, yerr=test2stds, capsize=10, label="2 class test")
        ax.errorbar(Cs, train6means, yerr=train6stds, capsize=10, label="6 class train")
        ax.errorbar(Cs, train2means, yerr=train2stds, capsize=10, label="2 class train")
        ax.legend()
        ax.grid(which='minor', axis='y')
        ax.grid(which='major', axis='x')
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("Accuracy")
        fig.show()
        save_figure(fig, os.path.join(self.image_path, "linearSVM"))

    def create_times_csv(self):
        times = pd.read_csv(self.times_file, sep=",")
        times_rounded = times.copy()
        times_rounded['mean'] = times_rounded.apply(lambda row: np.round(np.fromstring(row["mean"].
                                                    replace("[", "").replace("]", ""), sep=","), 4),
                                                    axis=1)
        times_rounded['std'] = times_rounded.apply(lambda row: np.round(np.fromstring(row["std"].
                                                   replace("[", "").replace("]", ""), sep=","),
                                                                        4), axis=1)
        times_rounded.to_csv(os.path.splitext(self.times_file)[0] + "_rounded.csv",
                             float_format='%.4f')
        x_data = times["diff"]
        y_data = times["acc_test_class6"]
        y_err = times["dev_test_class6"]
        z_data = times["mem"]

        s = (y_data >= .60) & (x_data <= 60)
        x_data = x_data[s]
        y_data = y_data[s]
        z_data = z_data[s]
        y_err = y_err[s]

        plt.errorbar(x_data, y_data, y_err, None, '.')
        plt.xlabel("time [s]")
        plt.ylabel("accuracy class6")
        plt.show()

        plt.clf()
        plt.errorbar(y_data, z_data, None, y_err, '.')
        plt.xlabel("accuracy class6")
        plt.ylabel("memory usage [MB]")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data, y_data, z_data)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('accuracy')
        ax.set_zlabel('memory usage [MB]')
        plt.show()
        times_publish = pd.DataFrame(times.copy())
        times_publish = times_publish.loc[:, ["model", "num_classes", "diff", "acc_test_class2",
                                              "dev_test_class2", "acc_test_class6",
                                              "dev_test_class6"]]
        model_ids = times_publish.loc[:, "model"]
        model_names = []
        model_origins = []
        for model in model_ids:
            splind = model.rfind("_")
            model_names.append(model[:splind])
            model_origins.append(model[splind + 1:])

        times_publish.loc[:, 'model_name'] = pd.Series(model_names, index=times_publish.index)
        times_publish.loc[:, 'model_origin'] = pd.Series(model_origins, index=times_publish.index)
        cols = times_publish.columns.tolist()
        times_publish = times_publish[cols[-2:] + cols[:-2]]
        times_publish.loc[:, ["acc_test_class2", "dev_test_class2",
                              "acc_test_class6", "dev_test_class6"]] = \
            times_publish.loc[:, ["acc_test_class2", "dev_test_class2",
                                  "acc_test_class6", "dev_test_class6"]] * 100
        times_publish = times_publish.sort_values("acc_test_class2", ascending=False)
        times_publish.loc[:, "arg_class2"] = np.arange(1, len(times_publish) + 1)
        times_publish = times_publish.sort_values("acc_test_class6", ascending=False)
        times_publish.loc[:, "arg_class6"] = np.arange(1, len(times_publish) + 1)
        names = ["dpn68b", "se_resnet50", "mnasnet0_5", "mnasnet1_0", "se_resnet152",
                 "resnext101_32x8d", "polynet", "shufflenet_v2_x1_0", "se_resnet101",
                 "densenet161", "inception_v3", "inceptionv3", "inceptionv4", "vgg19"]
        times_publish = times_publish.loc[
                        [True if name in names else False for name in times_publish["model_name"]],
                        :]
        times_publish = times_publish.loc[times_publish["num_classes"] == 1000, :]
        times_publish["diff"] = (times_publish["diff"] * 100) // 1 / 100
        times_publish["acc_test_class2"] = (times_publish["acc_test_class2"] * 100) // 1 / 100
        times_publish["dev_test_class2"] = (times_publish["dev_test_class2"] * 100) // 1 / 100
        times_publish["acc_test_class6"] = (times_publish["acc_test_class6"] * 100) // 1 / 100
        times_publish["dev_test_class6"] = (times_publish["dev_test_class6"] * 100) // 1 / 100

        times_publish = times_publish.drop(["model", "num_classes"], axis=1)
        times_publish = times_publish.rename(columns={'model_name': 'model name',
                                                      'model_origin': 'model origin',
                                                      'num_classes': 'num_features',
                                                      'diff': 'time [s]',
                                                      'acc_class2': '2 class accuracy',
                                                      'dev_class2': '2 class std',
                                                      'acc_class6': '6 class accuracy',
                                                      'dev_class6': '6 class std'})
        times_publish.to_csv("_publish".join(os.path.splitext(self.times_file)), index=False)

    def plot_magn_hist(self, values, n_bins):
        kde = KernelDensity(kernel='gaussian').fit(values.reshape(-1, 1))
        counts, bins = np.histogram(values, bins=n_bins)
        counts = counts / np.sum(counts)
        log_dens = kde.score_samples(bins.reshape(-1, 1))
        fig, ax = open_figure(constrained_layout=True)
        ax.fill(bins, np.exp(log_dens) / sum(np.exp(log_dens)), fc='#FF0000')
        ax.plot(bins[:-1], counts)
        ax.set_yscale("log")
        fig.show()
        return fig, ax

    def get_cloud_examples(self):
        data = self.preprocess_asim_feat_data()
        diff = 200
        width = diff//2
        cbhs = np.arange(diff//2, 6900, diff)
        hists = np.zeros(shape=(len(cbhs), 6))
        count = np.zeros(len(cbhs))
        c = 0
        for cbh in cbhs:
            sel = (data["CBH"] <= (cbh+diff//2)) & (data["CBH"] >= (cbh-diff//2))
            ser = pd.Series(index=np.arange(0, 6, 1), data=np.zeros(6), dtype=int)
            counts = data.loc[sel, "pred"].value_counts()
            ser.update(counts)
            hists[c] = ser.values/np.sum(ser.values)
            count[c] = np.sum(ser.values)
            c += 1
        fig, ax = open_figure(constrained_layout=True)
        class_name = "class_0"
        ax.bar(cbhs, hists[:, 0], width, label=self.classes_6.get(class_name).get("name"),
               color=self.classes_6.get(class_name).get("color"))
        b = 0
        for i in range(1, 6):
            b += hists[:, i-1]
            class_name = "class_"+str(i)
            ax.bar(cbhs, hists[:, i], width, label=self.classes_6.get(class_name).get("name"),
                   bottom=b, color=self.classes_6.get(class_name).get("color"))
        ax2 = ax.twinx()
        ax2.plot(cbhs, count/np.sum(count), color="C1")
        ax.set_xlabel("Cloud Base Height [m]")
        ax.set_ylabel("Occurence of predicted class based on CBH")
        ax2.set_ylabel("Occurence of measured CBH")
        ax2.set_ylim([0, ax2.get_ylim()[1]])
        ax.legend()
        save_figure(fig, self.image_path+"cloud/cbh_dist")
        fig.show()
        logger.info("Amount of images below 2000m: {}\nAmount of images in figure: {}".format(
            np.sum(data["CBH"] <= 2000), np.sum(count)))

    def segment_images(self):
        model, device = self.pp.set_model_and_device()
        ims_list = pd.read_csv(self.data_path + "interesting_ims.csv").values.ravel().tolist()
        file_list = [self.asim_path + s for s in ims_list]
        index_list = np.arange(0, len(ims_list)).tolist()
        n = 4
        data = pd.DataFrame()
        data["indices"] = [s for s in index_list for _ in range(n**2)]
        dataset = SegmentedLoader(file_list, index_list, splits=n)
        pb = ProgressBar(total=len(dataset), prefix='', suffix='', decimals=3, length=50, fill='=',
                         zfill='>')
        pb.generate_pbar(0)
        with torch.no_grad():
            for _, sample in enumerate(dataset, 0):
                inputs, index = sample
                mosaic, path = inputs
                mosaic = mosaic.to(device)
                output = model(mosaic)
                data.loc[data["indices"] == index, "paths"] = path
                data.loc[data["indices"] == index, self.feat_names] = output.cpu()
                pb.next()
        feats = data.loc[:, self.feat_names]
        preds = self.pp.predict_image_proba(feats)
        pred_class = np.argmax(preds, axis=1)
        data.loc[:, list(self.classes_6.keys())] = preds
        data.loc[:, "pred"] = pred_class
        data.to_hdf(self.data_path + "segments.hdf", "asim_segments", "w", format="fixed")

    def remove_image_parts(self):
        model, device = self.pp.set_model_and_device()
        ims_list = pd.read_csv(self.data_path + "interesting_ims.csv").values.ravel().tolist()
        file_list = [self.asim_path + s for s in ims_list]
        index_list = np.arange(0, len(ims_list)).tolist()
        n = 4
        data = pd.DataFrame()
        data["indices"] = [s for s in index_list for _ in range(n**2)]
        dataset = RemovedLoader(file_list, index_list, splits=n)
        pb = ProgressBar(total=len(dataset), prefix='', suffix='', decimals=3, length=50, fill='=',
                         zfill='>')
        pb.generate_pbar(0)
        with torch.no_grad():
            for _, sample in enumerate(dataset, 0):
                inputs, index = sample
                mosaic, path = inputs
                mosaic = mosaic.to(device)
                output = model(mosaic)
                data.loc[data["indices"] == index, "paths"] = path
                data.loc[data["indices"] == index, self.feat_names] = output.cpu()
                pb.next()
        feats = data.loc[:, self.feat_names]
        preds = self.pp.predict_image_proba(feats)
        pred_class = np.argmax(preds, axis=1)
        data.loc[:, list(self.classes_6.keys())] = preds
        data.loc[:, "pred"] = pred_class
        data.to_hdf(self.data_path + "removed_parts.hdf", "asim_removed", "w", format="fixed")

    def print_segm_ims(self, kind="removed"):
        if kind == "removed":
            file = "removed_parts"
            n = 4
        elif kind == "segmented":
            file = "segments"
            n = 4
        pred_data = self.preprocess_asim_feat_data()
        data: pd.DataFrame = pd.read_hdf(self.data_path + "{}.hdf".format(file))
        dataset = SegmentedLoader([], [], splits=n)
        l = dataset.size//n
        grid = determine_grid(n ** 2)
        pb = ProgressBar(total=len(data["indices"].unique()), prefix='', suffix='', decimals=3,
                         length=50, fill='=', zfill='>')
        pb.generate_pbar(0)
        for c in data["indices"].unique():
            path = data.loc[data["indices"] == c].iloc[0]["paths"]
            preds = np.array(data.loc[data["indices"] == c]["pred"])
            orig = pred_data.loc[pred_data["FILENAME"] == path.split(self.asim_path)[1], :]
            orig_pred = orig.iloc[0]["pred"]
            ims = self.get_file(path, None, dataset)[0]
            fig, main_ax = open_figure()
            main_ax.axis("off")
            ax_ar = fig.subplots(grid[1], grid[0])
            ax_ar = ax_ar.ravel()
            for i in range(n**2):
                pred = preds[i]
                x, y = np.divmod(i, 4)
                ax = ax_ar[i]
                im = np.array(ims[i, 1])
                ax.imshow(im[x*l:(x+1)*l, y*l:(y+1)*l], vmin=torch.min(ims), vmax=torch.max(ims))
                ax.set_xlabel(self.classes_6.get("class_{}".format(pred)).get("name"),
                              fontweight='normal' if pred == orig_pred else 'bold')
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.show()
            save_figure(fig, self.image_path + "asim/{}/".format(kind) +
                        os.path.splitext(os.path.split(path)[1])[0])
            pb.next()


class SegmentedLoader(AsimDataSet):
    def __init__(self, path_list: [], index_list, splits=16):
        self.splits = splits
        super(SegmentedLoader, self).__init__(path_list, index_list)

    def load_image(self, path):
        if not os.path.exists(path):
            return
        try:
            Image.open(path).verify()
        except (IOError, SyntaxError, UnidentifiedImageError):
            logger.error("Corrupted File")
            return None
        try:
            im = Image.open(path)
            trim = self.transform_nn(im)
        except Exception as e:
            logger.error(e)
            logger.error("Can't transform image")
        return trim, path

    def scaled_transform(self, window: int = 20):
        def image_weights():
            weights = {}
            for h in range(0, 750, 1):
                r = (h - 1) / 2
                x, y = (np.mgrid[:h, :h] - r)
                weight = 1 * ((x ** 2 + y ** 2) <= (r - window) ** 2)
                weights.update({h: weight})
            return weights

        def remove_upper_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im >= boundary] = boundary
            return im

        def remove_lower_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im <= boundary] = boundary
            return im

        def scale(im, weight):
            low = torch.min(im[np.where(weight)])
            high = torch.max(im[np.where(weight)])
            im = (im - low) / (high - low)
            return im

        def crop(im):
            return im[window:-window, window:-window]

        def crop_split(im):
            l = self.size // self.splits
            new_im = torch.zeros((self.splits**2, 3, self.size, self.size))
            for i, j in itertools.product(range(self.splits), range(self.splits)):
                new_im[j+i*self.splits, :, l*i:l*(i+1), l*j:l*(j+1)] =\
                    im[:, l*i:l*(i+1), l*j:l*(j+1)]
            return new_im

        weights = image_weights()
        transf = []
        # Transform to Tensor, remove additional dimensions
        transf.append(transforms.ToTensor())
        transf.append(transforms.Lambda(lambda x: x.squeeze()))
        # Crop by setting everything outside a disk to zero
        transf.append(transforms.Lambda(
            lambda x: 1. * x * torch.tensor(weights.get(x.shape[0]))
        ))
        # Remove upper and lower percentile to remove background and oversaturation
        transf.append(transforms.Lambda(
            lambda x: remove_lower_percentile(x, .5, weights.get(x.shape[0]))
        ))
        transf.append(transforms.Lambda(
            lambda x: remove_upper_percentile(x, 99.5, weights.get(x.shape[0]))
        ))
        # Scale min and max between 0 and 1
        transf.append(transforms.Lambda(
            lambda x: scale(x, weights.get(x.shape[0]))
        ))
        # Crop zeroed-elements out of image
        if window > 0:
            transf.append(transforms.Lambda(
                lambda x: crop(x)
            ))
        # GrayLevel -> RGB
        transf.append(transforms.Lambda(
            lambda x: torch.stack([torch.zeros(x.shape), x, torch.zeros(x.shape)])
        ))
        # Resize and Normalize to Specs of Neural network
        transf.append(transforms.Resize((self.size, self.size)))
        transf.append(transforms.Lambda(
            lambda x: crop_split(x)
        ))
        transf.append(transforms.Normalize(
            mean=self.mean, std=self.std
        ))
        trf = transforms.Compose(transf)
        return trf

class RemovedLoader(AsimDataSet):
    def __init__(self, path_list: [], index_list, splits=16):
        self.splits = splits
        super(RemovedLoader, self).__init__(path_list, index_list)

    def load_image(self, path):
        if not os.path.exists(path):
            return
        try:
            Image.open(path).verify()
        except (IOError, SyntaxError, UnidentifiedImageError):
            logger.error("Corrupted File")
            return None
        try:
            im = Image.open(path)
            trim = self.transform_nn(im)
        except Exception as e:
            logger.error(e)
            logger.error("Can't transform image")
        return trim, path

    def scaled_transform(self, window: int = 20):
        def image_weights():
            weights = {}
            for h in range(0, 750, 1):
                r = (h - 1) / 2
                x, y = (np.mgrid[:h, :h] - r)
                weight = 1 * ((x ** 2 + y ** 2) <= (r - window) ** 2)
                weights.update({h: weight})
            return weights

        def remove_upper_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im >= boundary] = boundary
            return im

        def remove_lower_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im <= boundary] = boundary
            return im

        def scale(im, weight):
            low = torch.min(im[np.where(weight)])
            high = torch.max(im[np.where(weight)])
            im = (im - low) / (high - low)
            return im

        def crop(im):
            return im[window:-window, window:-window]

        def remove_square(im):
            l = self.size // self.splits
            new_im = im.repeat(self.splits**2, 1, 1, 1)
            c = 0
            for i, j in itertools.product(range(self.splits), range(self.splits)):
                new_im[c, :, l*i:l*(i+1), l*j:l*(j+1)] = torch.zeros(3, l, l)
                c += 1
            return new_im

        weights = image_weights()
        transf = []
        # Transform to Tensor, remove additional dimensions
        transf.append(transforms.ToTensor())
        transf.append(transforms.Lambda(lambda x: x.squeeze()))
        # Crop by setting everything outside a disk to zero
        transf.append(transforms.Lambda(
            lambda x: 1. * x * torch.tensor(weights.get(x.shape[0]))
        ))
        # Remove upper and lower percentile to remove background and oversaturation
        transf.append(transforms.Lambda(
            lambda x: remove_lower_percentile(x, .5, weights.get(x.shape[0]))
        ))
        transf.append(transforms.Lambda(
            lambda x: remove_upper_percentile(x, 99.5, weights.get(x.shape[0]))
        ))
        # Scale min and max between 0 and 1
        transf.append(transforms.Lambda(
            lambda x: scale(x, weights.get(x.shape[0]))
        ))
        # Crop zeroed-elements out of image
        if window > 0:
            transf.append(transforms.Lambda(
                lambda x: crop(x)
            ))
        # GrayLevel -> RGB
        transf.append(transforms.Lambda(
            lambda x: torch.stack([torch.zeros(x.shape), x, torch.zeros(x.shape)])
        ))
        # Resize and Normalize to Specs of Neural network
        transf.append(transforms.Resize((self.size, self.size)))
        transf.append(transforms.Lambda(
            lambda x: remove_square(x)
        ))
        transf.append(transforms.Normalize(
            mean=self.mean, std=self.std
        ))
        trf = transforms.Compose(transf)
        return trf
