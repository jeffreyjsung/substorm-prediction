import itertools
import os
import pickle
from time import time

from common import save_figure, open_figure, best_network
import joblib
from console_progressbar import ProgressBar
from dataHandler.DataHandler import DataHandler
from dataHandler.Provider import Provider
from dataHandler.logger import logger
from dataHandler.datasets import AsimDataSet, OathDataSet

import numpy as np
import pandas as pd
import torch
import torchvision
import glob
from io import StringIO
import scipy.io as scio

import pretrainedmodels
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, classification_report


class PreProcessor(DataHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if os.path.exists(self.oath_clf_filename):
            self.clf = self.__load_oath_classifier()
        else:
            self.clf = None
        self.tv_models = []
        self.tv_model_names = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16",
                               "vgg16_bn", "vgg19", "vgg19_bn", "resnet18", "resnet34", "resnet50",
                               "resnet101", "resnet152", "squeezenet1_0", "squeezenet1_1",
                               "densenet121", "densenet169", "densenet161", "densenet201",
                               "inception_v3", "googlenet", "shufflenet_v2_x0_5",
                               "shufflenet_v2_x1_0", "mobilenet_v2", "resnext50_32x4d",
                               "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
                               "mnasnet0_5", "mnasnet1_0"]
        self.classifier_cross_validation_count = 10

    def proc_magn(self):
        logger.info("Processing available magn data.")
        years = np.arange(1993, 2021)
        months = np.arange(101, 113)
        i = 1
        pb = ProgressBar(total=len(years) * len(months), prefix='', suffix='', decimals=3,
                         length=50, fill='=', zfill='>')
        pb.print_progress_bar(0)
        for year in years:
            year = str(year)
            for month in months:
                month_file = pd.DataFrame()
                month = str(month)[-2:]
                files = glob.glob(self.magn_path + year + month + "*.txt")
                for file in files:
                    data = pd.read_fwf(file, widths=[4, 3, 3, 3, 3, 3, 11, 8, 8, 8, 8, 8, 8, 8, 8])
                    data = data[data.YYYY != "----"]
                    _t = data.columns[6::3]
                    for station in _t:
                        if station.find("Unnamed") >= 0:
                            continue
                        c = data.copy()
                        s = station[:3]
                        c["X"] = data[s + " X"]
                        c["Y"] = data[s + " Y"]
                        c["Z"] = data[s + " Z"]
                        c["LOC"] = s
                        c = c.drop(data.columns[6:], axis=1)
                        c["seconds"] = self.timestring_to_secs(c["HH"].astype(int),
                                                               c["MM.1"].astype(int),
                                                               c["SS"].astype(int))
                        c = c.drop(["YYYY", "MM", "HH", "MM.1", "SS"], axis=1)
                        c = c.rename(columns={"seconds": "SS"})
                        c = c.astype({'DD': 'int8', 'SS': 'int32',
                                      'X': 'float', 'Y': 'float', 'Z': 'float'})
                        c = c[["DD", "SS", "LOC", "X", "Y", "Z"]]
                        month_file = month_file.append(c)
                if not month_file.empty:
                    month_file.to_hdf(self.magn_path + year + month + ".hdf", "magn", "w",
                                      format="fixed")
                pb.print_progress_bar(i)
                i += 1

    def proc_ceil(self):
        logger.info("Processing available ceilometer data.")
        files = glob.glob(self.ceil_path + "*.tab")
        i = 1
        pb = ProgressBar(total=len(files), prefix='', suffix='', decimals=3,
                         length=50, fill='=', zfill='>')
        pb.print_progress_bar(0)
        for file in files:
            f = open(file, "r")
            t = f.read()
            start = t.rfind("Date/Time	CBH [m]")
            data = pd.read_csv(StringIO(t[start:]), delimiter="\t")
            data = data.rename(columns={"Date/Time": "DATE", "CBH [m]": "CBH"})
            data["CBH"] = np.clip(data["CBH"].fillna(-1).astype(int), -1, np.iinfo(np.int16).max).astype(
                np.int16)
            date = os.path.splitext(file)[0].split("_")[-1]
            year = date.split("-")[0]
            month = date.split("-")[1]
            data["DD"] = data["DATE"].str.split("T", expand=True)[0].str.split("-", expand=True)[2] \
                .astype(np.int8)
            hours = data["DATE"].str.split("T", expand=True)[1].str.split(":", expand=True)[0] \
                .astype(np.int32)
            minutes = data["DATE"].str.split("T", expand=True)[1].str.split(":", expand=True)[1] \
                .astype(np.int32)
            data["SS"] = hours * 3600 + minutes * 60
            data = data.astype({'SS': 'int32'})
            data["LOC"] = "NYA"
            data = data[["DD", "SS", "LOC", "CBH"]]
            if not data.empty:
                data.to_hdf(self.ceil_path + year + month + ".hdf", "ceil", "w", format="fixed")
            pb.print_progress_bar(i)
            i += 1

    def set_model_and_device(self, device = None):
        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        logger.info("Device set to {}".format(device))
        logger.info("loading pretrained torchvision model")
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.eval()
        model.to(device)
        return model, device

    def proc_asim(self, batch_size=64):
        logger.info("Processing available asim data.")
        model, device = self.set_model_and_device()
        logger.info("reading image index file")
        full_list = Provider.asim_links_file(self)
        for year in full_list["YYYY"].unique():
            for month in full_list[full_list["YYYY"] == year]["MM"].unique():
                month_str = str(month)
                if len(month_str) == 1:
                    month_str = "0" + month_str
                logger.info("Currently processing: {}-{}".format(year, month_str))
                monthly_file_name = self.asim_path + str(year) + str(month_str) + ".hdf"
                if os.path.exists(monthly_file_name):
                    logger.info("Already processed")
                    continue
                monthly_list = full_list.loc[(full_list["YYYY"] == year) &
                                             (full_list["MM"] == month), :]\
                    .copy().reset_index(drop=True)
                colums = ["LOC", "WVL"]
                kinds = monthly_list.drop_duplicates(subset=colums)
                N = np.ceil(len(monthly_list)/batch_size) + len(kinds) -1
                pb = ProgressBar(total=N, prefix='', suffix='', decimals=3, length=50, fill='=',
                                 zfill='>')
                pb.print_progress_bar(0)
                mf_data = []
                for _, kind in kinds.iterrows():
                    # year, month, wvl, loc, batch / tot_batch: 2011, 11, 5577, NYA4, 41/62
                    wvl = kind["WVL"]
                    loc = kind["LOC"]
                    kind_list = monthly_list.loc[(monthly_list["WVL"] == wvl) &
                                                 (monthly_list["LOC"] == loc), :]
                    file_list = list(kind_list.loc[:, "FILENAME"])
                    file_list = [self.asim_path + s for s in file_list]
                    index_list = list(kind_list.index)
                    data = AsimDataSet(file_list, index_list)
                    data_loader = torch.utils.data.dataloader.DataLoader(data, shuffle=False,
                                                                         batch_size=batch_size,
                                                                         collate_fn=self.my_collate)
                    with torch.no_grad():
                        try:
                            for i_batch, sample in enumerate(data_loader, 0):
                                sample_batched = sample[0]
                                index = sample[1].numpy()
                                flat = sample_batched[0].to(device).squeeze()
                                input = sample_batched[1].to(device)
                                hist = sample_batched[2].to(device)
                                paths = sample_batched[3]
                                output = model(input)
                                feats = self.__extract_features(hist, flat, device)
                                feats[self.feat_names] = output.cpu()
                                feats.index = index
                                mf_data.append(feats)
                                pb.next()
                        except Exception as e:
                            logger.error(e)
                            logger.info(i_batch)
                            logger.info(paths)
                            logger.info(index)
                            logger.info(kind)
                            logger.info(kinds)
                            logger.info("year, month, wvl, loc: {},{},{},{}".format(year, month, wvl, loc))
                            return data
                if len(mf_data) == 0:
                    continue
                features = pd.concat(mf_data)
                monthly_list = pd.concat([monthly_list, features], axis=1)
                monthly_list["SS"] = monthly_list.apply(
                    lambda row: int(row["HH"]) * 3600 + int(row["mm"]) * 60 + int(row["SS"]),
                    axis=1)
                monthly_list = monthly_list.drop(["YYYY", "MM", "HH", "mm"], axis=1)
                monthly_file = monthly_list.astype({"DD": np.int8, "SS": np.int32,
                                                    "WVL": np.int16})
                monthly_file.to_hdf(monthly_file_name, "asim", "w", format="fixed")

    def proc_asim_cali(self):
        availability = Provider.asim_availability_file(self)
        pb = ProgressBar(total=len(availability), prefix='', suffix='', decimals=3,
                         length=50, fill='=', zfill='>')
        pb.print_progress_bar(0)
        for _, avs in availability.groupby(["YYYY", "MM"]).size().reset_index().iterrows():
            year = avs["YYYY"]
            month = avs["MM"]
            datas = dict()
            for _, row in availability.loc[(availability["YYYY"] == year) &
                                           (availability["MM"] == month)].iterrows():
                loc = row["LOC"]
                wvl = row["WVL"]
                year = row["YYYY"]
                month = row["MM"]
                day = row["DD"]
                ymd = "".join([year, month, day])
                path = "/".join([loc, wvl, year, ymd])
                fname = "_".join([loc, ymd, wvl, "cal"]) + ".dat"
                fullpath = "/".join([self.asim_path, path, fname])
                if not os.path.exists(fullpath):
                    pb.next()
                    continue
                dd = scio.readsav(fullpath)
                nd = dict()
                for key in dd.keys():
                    data = dd.get(key)
                    nd.update({key: data})
                datas.update({"_".join([ymd, loc, wvl]): nd})
                pb.next()
            if len(datas) > 0:
                self.save_obj(datas, self.asim_path+"calibrations_{}{:02d}.dat".
                              format(year, int(month)))

    def create_log_header(self):
        file_object = open(self.times_file, "w")
        file_object.write(
            "{},{},{},{},{},{},{},{}\n".format("model", "key", "num_classes", "size",
                                               "mean", "std", "diff", "mem"))
        file_object.close()

    @staticmethod
    def check_skip(skip_list, name):
        return any(x in name.lower() for x in skip_list)

    def load_tv_models(self):
        self.tv_models = [torchvision.models.alexnet(pretrained=True),
                          torchvision.models.vgg11(pretrained=True),
                          torchvision.models.vgg11_bn(pretrained=True),
                          torchvision.models.vgg13(pretrained=True),
                          torchvision.models.vgg13_bn(pretrained=True),
                          torchvision.models.vgg16(pretrained=True),
                          torchvision.models.vgg16_bn(pretrained=True),
                          torchvision.models.vgg19(pretrained=True),
                          torchvision.models.vgg19_bn(pretrained=True),
                          torchvision.models.resnet18(pretrained=True),
                          torchvision.models.resnet34(pretrained=True),
                          torchvision.models.resnet50(pretrained=True),
                          torchvision.models.resnet101(pretrained=True),
                          torchvision.models.resnet152(pretrained=True),
                          torchvision.models.squeezenet1_0(pretrained=True),
                          torchvision.models.squeezenet1_1(pretrained=True),
                          torchvision.models.densenet121(pretrained=True),
                          torchvision.models.densenet169(pretrained=True),
                          torchvision.models.densenet161(pretrained=True),
                          torchvision.models.densenet201(pretrained=True),
                          torchvision.models.inception_v3(pretrained=True),
                          torchvision.models.googlenet(pretrained=True),
                          torchvision.models.shufflenet_v2_x0_5(pretrained=True),
                          torchvision.models.shufflenet_v2_x1_0(pretrained=True),
                          torchvision.models.mobilenet_v2(pretrained=True),
                          torchvision.models.resnext50_32x4d(pretrained=True),
                          torchvision.models.resnext101_32x8d(pretrained=True),
                          torchvision.models.wide_resnet50_2(pretrained=True),
                          torchvision.models.wide_resnet101_2(pretrained=True),
                          torchvision.models.mnasnet0_5(pretrained=True),
                          torchvision.models.mnasnet1_0(pretrained=True)]

    def evaluate_network_performances(self, mode="both", ignore=[]):
        if not os.path.exists(self.times_file):
            self.create_log_header()
        if (mode == "both") or (mode == "cadene"):
            cadene_model_names = pretrainedmodels.model_names
            for model_name in cadene_model_names:
                logger.info(model_name)
                if self.check_skip(ignore, model_name):
                    logger.info("skipping")
                    continue
                inf = pretrainedmodels.pretrained_settings[model_name]
                for key in inf.keys():
                    num_classes = inf[key]['num_classes']
                    size = inf[key]["input_size"][1]
                    mean = inf[key]["mean"]
                    std = inf[key]["std"]
                    logger.info(num_classes)
                    model = getattr(pretrainedmodels, model_name)(num_classes=num_classes,
                                                                  pretrained=key)
                    self.extract_oath_features(model, model_name, key, "cadene", num_classes, size,
                                               mean, std)
        if (mode == "both") or (mode == "torchvision"):
            self.load_tv_models()
            for model, model_name in zip(self.tv_models, self.tv_model_names):
                logger.info(model_name)
                if self.check_skip(ignore, model_name):
                    logger.info("skipping")
                    continue
                if model_name == "inception_v3":
                    size = 299
                else:
                    size = 224
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                ll = model._modules[list(model._modules)[-1]]
                if hasattr(ll, 'out_features'):
                    num_classes = ll.out_features
                else:
                    for l in reversed(ll):
                        if hasattr(l, 'out_features'):
                            num_classes = l.out_features
                            break
                        if hasattr(l, 'out_channels'):
                            num_classes = l.out_channels
                            break
                if num_classes == 1000:
                    key = "imagenet"
                else:
                    key = "imagenet+background"
                self.extract_oath_features(model, model_name, key, "torchvision", num_classes, size,
                                           mean, std)

    def evaluate_network_accuracies(self):
        np.random.seed(42)
        classifications = self.read_oath_labels()
        times = pd.read_csv(self.times_file, sep=",")
        if not times.columns.__contains__("acc_test_6"):
            classes = ["class2", "class6"]
            files = glob.glob(os.path.join(self.torch_path, "*.csv"))
            for c in classes:
                for file in files:
                    if file.__contains__("times"):
                        continue
                    nl = file.split("oath_features_")[1].split(".csv")[0].split("_")
                    name = "_".join(nl[:-2])
                    key = nl[-2]
                    source = nl[-1]
                    logger.info("{}_{}_{}".format(name, key, source))
                    features = pd.read_csv(file, index_col=0)
                    alpha = 0.03
                    clf = RidgeClassifier(normalize=True, alpha=alpha)
                    accs_test, _, accs_train, _, test_mat, train_mat = \
                    self.test_clf(clf, features, classifications, c)
                    acc_test = np.mean(accs_test)
                    acc_train = np.mean(accs_train)
                    dev_test = np.std(accs_test)
                    dev_train = np.std(accs_train)
                    times.loc[np.array(times["model"] == "_".join([name, source])) & np.array(
                        times["key"] == key), ["acc_test_" + c, "dev_test_" + c, "acc_train_" + c,
                                               "dev_train_" + c]] = [acc_test, dev_test,
                                                                     acc_train, dev_train]
            times.to_csv(self.times_file, index=False)

    def proc_oath(self):
        self.evaluate_network_performances()
        self.evaluate_network_accuracies()
        self.evaluate_different_svms()
        c = self.best_hyperparameter()
        clf = self.fit_oath_features(c)
        self.save_oath_classifier(clf)

    def predict_image_proba(self, features):
        if self.clf is None:
            logger.error("No saved classifier found. Please train and save it first.")
            return
        return self.clf.predict_proba(features)

    def extract_oath_features(self, model, model_name, key, source, num_classes, size, mean, std):
        name = self.torch_path+"oath_features_{}_{}_{}.csv".format(model_name, key,
                                                                             source)
        if os.path.exists(name):
            return
        feat_names = ["feat_" + str(i) for i in range(num_classes)]
        oath_image_path = self.oath_path+"images/cropped_scaled_rotated/"
        file_list = sorted(os.listdir(oath_image_path))
        index_list = [int(name.split("_")[0]) for name in file_list]
        file_list = [oath_image_path + file for file in file_list]
        data = OathDataSet(file_list, index_list, size, mean, std)
        data_loader = torch.utils.data.dataloader.DataLoader(data, shuffle=False, batch_size=64)
        features = []
        if torch.cuda.is_available():
            nvmlInit()
            device = "cuda:0"
        else:
            device = "cpu"
        model.to(device)
        model.eval()
        with torch.no_grad():
            since = time()
            for i_batch, sample in enumerate(data_loader, 0):
                sample_batched = sample[0]
                index = sample[1].numpy()
                input = sample_batched[0].to(device)
                output = model(input)
                out = output.cpu().squeeze().numpy()
                feats = pd.DataFrame(out, index=index, columns=feat_names)
                features.append(feats)
                if torch.cuda.is_available():
                    h = nvmlDeviceGetHandleByIndex(0)
                    info = nvmlDeviceGetMemoryInfo(h)
                    mem = info.used / 1024 / 1024
                else:
                    mem = -1
                logger.info(str(i_batch + 1) + " / " + str(len(data_loader)))
        diff = time() - since
        logger.info("time used for feature extraction: {}s".format(diff))
        file_object = open(self.times_file, "a")
        file_object.write("{},{},{},{},\"{}\",\"{}\",{},{}\n"
                          .format("{}_{}".format(model_name, source), key, num_classes, size, mean,
                                  std, diff, mem))
        file_object.close()
        features = pd.concat(features)
        features.to_csv(name)
        return features

    def evaluate_different_svms(self):
        self.evaluate_RBF()
        self.evaluate_linear()

    def evaluate_RBF(self):
        cs = np.logspace(-1, 3, 13)
        gs = np.logspace(-6, -2, 13)
        features = pd.read_csv(self.torch_path+best_network, index_col=0)
        np.random.seed(42)
        classifications = self.read_oath_labels()
        rbf_accs = {}
        if os.path.exists(self.rbf_accs_file):
            f = open(self.rbf_accs_file, "rb")
            accs_loaded = pickle.load(f)
            f.close()
            rbf_accs.update(accs_loaded)
        for c, g in itertools.product(cs, gs):
            if rbf_accs.get((c, g)) is not None:
                continue
            logger.info((c, g))
            clf = svm.SVC(kernel='rbf', probability=True, gamma=g, C=c, verbose=False)
            test_accs_6, test_accs_2, train_accs_6, train_accs_2, test_mat, train_mat = \
                self.test_clf(clf, features, classifications)
            test6means = np.mean(test_accs_6)
            test2means = np.mean(test_accs_2)
            train6means = np.mean(train_accs_6)
            train2means = np.mean(train_accs_2)
            test6stds = np.std(test_accs_6)
            test2stds = np.std(test_accs_2)
            train6stds = np.std(train_accs_6)
            train2stds = np.std(train_accs_2)
            rbf_accs.update({(c, g): {
                "test6means": test6means,
                "test2means": test2means,
                "train6means": train6means,
                "train2means": train2means,
                "test6stds": test6stds,
                "test2stds": test2stds,
                "train6stds": train6stds,
                "train2stds": train2stds,
            }})
            f = open(self.rbf_accs_file, "wb")
            pickle.dump(rbf_accs, f)
            f.close()

    def evaluate_linear(self):
        features = pd.read_csv(self.torch_path + best_network, index_col=0)
        classifications = self.read_oath_labels()
        np.random.seed(42)
        linear_accs = {}
        l = 9
        Cs = np.logspace(-5, 0, l)
        Cs = np.append(Cs, np.logspace(-4.3, -3.3, l))
        if os.path.exists(self.linear_accs_file):
            f = open(self.linear_accs_file, "rb")
            accs_loaded = pickle.load(f)
            f.close()
            linear_accs.update(accs_loaded)
        for c in Cs:
            if linear_accs.get(c) is not None:
                continue
            clf = svm.SVC(kernel='linear', probability=True, C=c)
            test_accs_6, test_accs_2, train_accs_6, train_accs_2, test_mat, train_mat = \
                self.test_clf(clf, features, classifications, "class6")
            test6means = np.mean(test_accs_6)
            test2means = np.mean(test_accs_2)
            train6means = np.mean(train_accs_6)
            train2means = np.mean(train_accs_2)
            test6stds = np.std(test_accs_6)
            test2stds = np.std(test_accs_2)
            train6stds = np.std(train_accs_6)
            train2stds = np.std(train_accs_2)
            linear_accs.update({c: {
                "test6means": test6means,
                "test2means": test2means,
                "train6means": train6means,
                "train2means": train2means,
                "test6stds": test6stds,
                "test2stds": test2stds,
                "train6stds": train6stds,
                "train2stds": train2stds,
            }})
            f = open(self.linear_accs_file, "wb")
            pickle.dump(linear_accs, f)
            f.close()

    def best_hyperparameter(self):
        if not os.path.exists(self.linear_accs_file):
            return
        f = open(self.linear_accs_file, "rb")
        accs = pickle.load(f)
        f.close()
        c = -1
        acc = 0
        for v in accs:
            a = accs.get(v)
            if a.get("test6means") > acc:
                c = v
                acc = a.get("test6means")
        return c

    def read_oath_labels(self):
        return pd.read_csv(self.oath_path+"/classifications/classifications.csv", header=16)

    def fit_oath_features(self, c):
        features = pd.read_csv(self.torch_path + best_network, index_col=0)
        classifications = self.read_oath_labels()
        clsf = classifications["class6"]

        ndata = len(features.index)
        dist = clsf.value_counts().sort_index()/ndata
        id = np.arange(0, ndata//10*3, 1)
        diff = ndata
        best = id
        for i in range(ndata):
            id = np.remainder(id + 1, ndata)
            ldiff = np.sum(np.abs(clsf[id].value_counts().sort_index()/len(id) - dist))
            if ldiff < diff:
                diff = ldiff
                best = id

        x_test = features.iloc[best]
        x_train = features.loc[~features.index.isin(best)]
        y_test = clsf.iloc[best]
        y_train = clsf.loc[~features.index.isin(best)]

        best_dist = y_test.value_counts().sort_index()/len(id)
        train_dist = y_train.value_counts().sort_index()/len(y_train)

        fig, ax = open_figure(constrained_layout=True)
        ax.bar(best_dist.index-.2, best_dist, width=0.2, label="testing data")
        ax.bar(dist.index, dist, width=0.2, label="full distribution")
        ax.bar(train_dist.index+.2, train_dist, width=0.2, label="training data")
        ax.legend()
        ax.set_xlabel("class number")
        fig.show()
        save_figure(fig, self.image_path + "training distribution")

        clf = svm.SVC(kernel='linear', probability=True, C=c, verbose=True)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_mat = confusion_matrix(y_test, y_pred)
        train_mat = confusion_matrix(y_train, clf.predict(x_train))
        test_acc_6 = np.sum(y_pred == y_test) / len(y_pred)
        train_acc_6 = np.sum(clf.predict(x_train) == y_train) / len(y_train)
        test_acc_2 = (np.sum(test_mat[:3, :3]) + np.sum(test_mat[3:, 3:])) / len(y_pred)
        train_acc_2 = (np.sum(train_mat[:3, :3]) + np.sum(train_mat[3:, 3:])) / len(y_train)
        logger.info("Testing Conf mat\n" + test_mat.__str__())
        logger.info("Training Conf mat\n" + train_mat.__str__())
        logger.info("Testing Accuracy 6 class: {:.4f}".format(test_acc_6 * 100))
        logger.info("Testing Accuracy 2 class: {:.4f}".format(test_acc_2 * 100))
        logger.info("Training Accuracy 6 class: {:.4f}".format(train_acc_6 * 100))
        logger.info("Training Accuracy 2 class: {:.4f}".format(train_acc_2 * 100))
        logger.info("Classification report testing 6 class:\n" +
                    classification_report(y_test, y_pred))
        logger.info("Classification report training 2 class:\n" +
                    classification_report(y_train, clf.predict(x_train)))
        logger.info("Classification report testing 2 class:\n" +
                    classification_report(y_test > 2, y_pred > 2))
        logger.info("Classification report testing 2 class:\n" +
                    classification_report(y_train > 2, clf.predict(x_train) > 2))
        return clf

    def test_clf(self, clf, features, classifications, classes):
        ndata = len(features.index)
        ids = np.arange(0, ndata, 1)
        idxs = np.zeros(shape=(self.classifier_cross_validation_count, ndata))
        for i in range(self.classifier_cross_validation_count):
            idxs[i, :] = np.roll(ids, -int(ndata//self.classifier_cross_validation_count*i))
        train_accs_6 = np.zeros(self.classifier_cross_validation_count)
        test_accs_6 = np.zeros(self.classifier_cross_validation_count)
        if classes == "class6":
            train_accs_2 = np.zeros(self.classifier_cross_validation_count)
            test_accs_2 = np.zeros(self.classifier_cross_validation_count)
        cnt = 0
        for idx in idxs:
            logger.info("{}/{}".format(cnt+1, self.classifier_cross_validation_count))
            ntrain = int(np.round(0.7 * ndata))
            idx_train = idx[0:ntrain]
            idx_test = idx[ntrain:]
            X_train = features.iloc[idx_train]
            y_train = classifications[classes][idx_train]
            X_test = features.iloc[idx_test]
            y_test = classifications[classes][idx_test]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            test_mat = confusion_matrix(y_test, y_pred)
            train_mat = confusion_matrix(y_train, clf.predict(X_train))
            test_acc_6 = np.sum(y_pred == y_test)/len(y_pred)
            train_acc_6 = np.sum(clf.predict(X_train) == y_train)/len(y_train)
            if classes == "class6":
                test_acc_2 = (np.sum(test_mat[:3, :3])+np.sum(test_mat[3:, 3:]))/len(y_pred)
                train_acc_2 = (np.sum(train_mat[:3, :3])+np.sum(train_mat[3:, 3:]))/len(y_train)
            test_accs_6[cnt] = test_acc_6
            train_accs_6[cnt] = train_acc_6
            if classes == "class6":
                test_accs_2[cnt] = test_acc_2
                train_accs_2[cnt] = train_acc_2
            cnt += 1
        logger.info("Testing Conf mat\n" + test_mat.__str__())
        logger.info("Training Conf mat\n" + train_mat.__str__())
        logger.info("Mean Testing Accuracy of six class: {:.4f}".format(np.mean(test_accs_6)))
        logger.info("Mean Testing Accuracy of six class std: {:.4f}".format(np.std(test_accs_6)))
        if classes == "class6":
            logger.info("Mean Testing Accuracy of two class: {:.4f}".format(np.mean(test_accs_2)))
            logger.info("Mean Testing Accuracy of two class std: {:.4f}".format(np.std(test_accs_2)))
        logger.info("Testing Accuracy of last run: {:.4f}".format(test_accs_6[-1] * 100))
        if classes == "class6":
            logger.info("Testing Accuracy nl/no_nl of last run: {:.4f}".format(test_accs_2[-1] * 100))
            logger.info("Mean Training Accuracy of six class: {:.4f}".format(np.mean(train_accs_6)))
        logger.info("Mean Training Accuracy of six class std: {:.4f}".format(np.std(train_accs_6)))
        if classes == "class6":
            logger.info("Mean Training Accuracy of two class: {:.4f}".format(np.mean(train_accs_2)))
            logger.info("Mean Training Accuracy of two class std: {:.4f}".format(np.std(train_accs_2)))
        logger.info("Training Accuracy of last run: {:.4f}".format(train_accs_6[-1] * 100))
        if classes == "class6":
            logger.info("Training Accuracy nl/no_nl of last run: {:.4f}".format(train_accs_2[-1] * 100))
        if classes == "class6":
            return test_accs_6, test_accs_2, train_accs_6, train_accs_2, test_mat, train_mat
        else:
            return test_accs_6, None, train_accs_6, None, test_mat, train_mat

    def save_oath_classifier(self, clf):
        joblib.dump(clf, self.oath_clf_filename)

    def __load_oath_classifier(self):
        clf = joblib.load(self.oath_clf_filename)
        return clf

    @staticmethod
    def __define_weights():
        weights = {}
        for h in range(1, 525, 2):
            r = (h - 1) / 2
            x, y = (np.mgrid[:h, :h] - r)

            weight = 1 * ((x ** 2 + y ** 2) <= (r + 0.5) ** 2)
            weight_here = {h: weight}
            weights.update(weight_here)
        return weights

    @staticmethod
    def __extract_features(hist, flat, device):
        bins = torch.arange(0, hist.shape[1]).unsqueeze(1).repeat(1, hist.shape[0]).to(device)
        prob = torch.div(hist.T, torch.sum(hist, axis=1)).T
        mi = torch.min(flat, axis=1).values
        ma = torch.max(flat, axis=1).values
        median = torch.median(flat, axis=1).values
        mean = torch.mean(flat, axis=1)

        variance = torch.sum((bins - mean).T ** 2 * prob, axis=1)
        skewness = torch.sqrt(variance) ** -3 * torch.sum((bins - mean).T ** 3 * prob, axis=1)
        kurtosis = torch.sqrt(variance) ** -4 * torch.sum((bins - mean).T ** 4 * prob, axis=1) - 3
        entropy = -torch.nansum(torch.log(prob) * prob, axis=1)
        energy = torch.sum(prob ** 2, axis=1)
        return pd.DataFrame(
            {"MIN": mi.cpu(), "MAX": ma.cpu(), "MEDIAN": median.cpu(), "MEAN": mean.cpu(),
             "VARIANCE": variance.cpu(), "SKEWNESS": skewness.cpu(), "KURTOSIS": kurtosis.cpu(),
             "ENTROPY": entropy.cpu(), "ENERGY": energy.cpu()})
