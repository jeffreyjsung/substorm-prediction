import os
import pickle

from datetime import datetime

import torch
from matplotlib import colors
import matplotlib.pyplot as plt

import numpy as np

MAGN_PATH = "tame/data/magn/"
CEIL_PATH = "tame/data/ceil/"
ASIM_PATH = "tame/data/asim/"
OATH_PATH = "tame/data/oath/"
DATA_PATH = "tame/data/other/"
TORCH_PATH = "tame/data/torch/"
IMAGE_PATH = "tame/data/images/"
CONFIG_FILE = "tame/dataHandler/config.ini"


class DataHandler:
    magn_path = ""
    ceil_path = ""
    asim_path = ""
    oath_path = ""
    data_path = ""
    torch_path = ""
    image_path = ""
    config_file = ""
    ceil_index_dl = "https://doi.pangaea.de/10.1594/PANGAEA.880300?format=textfile&charset=UTF-8"
    asim_index_dl = "http://tid.uio.no/plasma/aurora/avail.csv"
    oath_dl = "http://tid.uio.no/plasma/oath/oath_v1.1_20181026.tgz"
    classes_6 = {
        "class_0": {"name": "arc", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[0]),
                    "marker": "o"},
        "class_1": {"name": "diffuse", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[1]),
                    "marker": "o"},
        "class_2": {"name": "discrete", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[2]),
                    "marker": "o"},
        "class_3": {"name": "cloud", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[3]),
                    "marker": "x"},
        "class_4": {"name": "moon", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[4]),
                    "marker": "x"},
        "class_5": {"name": "clear sky", "color": colors.rgb2hex(plt.get_cmap("Set1").colors[6]),
                    "marker": "x"}
    }

    classes_2 = {
        "class_0": {"name": "aurora"},
        "class_1": {"name": "no aurora"}
    }

    feat_names = ["feat_" + str(i) for i in range(1000)]

    def __init__(self, **kwargs):
        self.magn_path = kwargs.get("magn_path", MAGN_PATH)
        self.ceil_path = kwargs.get("ceil_path", CEIL_PATH)
        self.asim_path = kwargs.get("asim_path", ASIM_PATH)
        self.oath_path = kwargs.get("oath_path", OATH_PATH)
        self.data_path = kwargs.get("analyzer_data_path", DATA_PATH)
        self.image_path = kwargs.get("image_path", IMAGE_PATH)
        self.torch_path = kwargs.get("torch_path", TORCH_PATH)

        self._check_create_path(self.magn_path)
        self._check_create_path(self.ceil_path)
        self._check_create_path(self.asim_path)
        self._check_create_path(self.oath_path)
        self._check_create_path(self.torch_path)
        self._check_create_path(self.data_path)
        self._check_create_path(self.data_path+"reductions/")
        self._check_create_path(self.image_path)
        folders = ["asim", "magn", "ceil", "test", "asim/segmented"]
        for folder in folders:
            self._check_create_path(self.image_path+folder)

        self.oath_feat_filename = self.torch_path+"oath_features.csv"
        self.oath_clf_filename = self.torch_path+"oath.clf"
        self.times_file = self.torch_path+"times.csv"
        self.rbf_accs_file = self.torch_path+"accs_SVM_RBF.pkl"
        self.linear_accs_file = self.torch_path+"accs_SVM_linear.pkl"
        self.magn_split = self.data_path + "magn_split.csv"

    @staticmethod
    def _check_create_path(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_obj(obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def date_iterator(date_start: datetime, date_end: datetime, diff):
        curr_date = date_start
        while curr_date <= date_end:
            yield curr_date
            curr_date += diff

    @staticmethod
    def my_collate(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    @staticmethod
    def timestring_to_secs(hours, minutes, seconds):
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def row_times_to_seconds(row):
        return datetime(year=int(row["YYYY"]), month=int(row["MM"]),
                        day=int(row["DD"]),
                        hour=int(row["SS"]) // 3600,
                        minute=np.remainder(int(row["SS"]), 3600) // 60,
                        second=np.remainder(int(row["SS"]), 60)).timestamp()
