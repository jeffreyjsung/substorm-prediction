import calendar
import os
from datetime import datetime
from time import time
from console_progressbar import ProgressBar
import glob
import pandas as pd
from dataHandler.DataHandler import DataHandler
from joblib import Parallel, delayed
from dataHandler.logger import logger
import numpy as np


class Provider(DataHandler):
    def __init__(self, **kwargs):
        super(Provider, self).__init__(**kwargs)

    def get_magn(self, **kwargs):
        location = kwargs.get("location", None)
        date_start = kwargs.get("date_start", datetime.today())
        date_end = kwargs.get("date_end", datetime.today())
        return self.__get_files(data_location=self.magn_path, location=location,
                                date_start=date_start, date_end=date_end)

    def get_ceil(self, **kwargs):
        location = kwargs.get("location", None)
        date_start = kwargs.get("date_start", datetime.today())
        date_end = kwargs.get("date_end", datetime.today())
        return self.__get_files(data_location=self.ceil_path, location=location,
                                date_start=date_start, date_end=date_end)

    def get_asim(self, **kwargs):
        location = kwargs.get("location", None)
        date_start = kwargs.get("date_start", datetime.today())
        date_end = kwargs.get("date_end", datetime.today())
        return self.__get_files(data_location=self.asim_path, location=location,
                                date_start=date_start, date_end=date_end)

    def __get_files(self, data_location,  **kwargs):
        location = kwargs.get("location", None)
        date_start = kwargs.get("date_start", datetime.today())
        date_end = kwargs.get("date_end", datetime.today())
        files = sorted(glob.glob(data_location + "19*.hdf") + glob.glob(data_location + "20*.hdf"))
        out = pd.DataFrame()
        for file in files:
            name = os.path.splitext(file.split(data_location)[1])[0][:6]
            year = int(name[:-2])
            month = int(name[-2:])
            last_day = calendar.monthrange(year, month)[1]
            tot_days = (date_end - date_start).days
            if datetime(year=year, month=month, day=last_day) < date_start or \
                    datetime(year=year, month=month, day=1) > date_end:
                continue
            data = pd.read_hdf(file)
            data["FLOC"] = data["LOC"]
            data["LOC"] = data.apply(lambda row: row["LOC"][:-1] if len(row["LOC"]) == 4
                else row["LOC"], axis=1)
            if location:
                data = data[data["LOC"] == location]
            if tot_days < last_day:
                data = data[data["DD"].isin(np.arange(date_start.day, date_end.day+1).tolist())]
            if data.columns.__contains__("HH"):
                data["SS"] = data.apply(lambda row: int(row["HH"]) * 3600 + int(row["mm"]) * 60 +
                                                    int(row["SS"]), axis=1)
                data = data.drop(["HH", "mm"], axis=1)
            if not data.columns.__contains__("MM"):
                data.insert(0, "MM", month)
            if not data.columns.__contains__("YYYY"):
                data.insert(0, "YYYY", year)
            out = out.append(data)
        if len(out) > 0:
            columns = out.columns
            first_columns = ["YYYY", "MM", "DD", "SS", "LOC"]
            order = first_columns + [col for col in columns if col not in first_columns]
            out = out[order]
        return out

    def asim_availability_file(self):
        return pd.DataFrame(pd.read_hdf(self.asim_path+"asim_availability.hdf"))

    def asim_links_file(self):
        files = glob.glob(self.asim_path+"asim_links_*.hdf")
        dfl = []
        for file in files:
            dfl.append(pd.read_hdf(file))
        return pd.concat(dfl)

    @staticmethod
    def date_in_timeframe(date, date_start, date_end):
        return (date_start <= date) & (date <= date_end)

    def asim_calibs(self, **kwargs):
        location = kwargs.get("location", None)
        date_start = kwargs.get("date_start", datetime.today())
        date_end = kwargs.get("date_end", datetime.today())
        files = sorted(glob.glob(self.asim_path + "calibrations_*.dat"))
        out = dict()
        for file in files:
            name = os.path.splitext(file.split(self.asim_path + "calibrations_")[1])[0][:6]
            year = int(name[:-2])
            month = int(name[-2:])
            last_day = np.min([calendar.monthrange(year, month)[1], date_end.day])
            first_day = np.max([1, date_start.day])
            if not (self.date_in_timeframe(datetime(year=year, month=month, day=last_day),
                                           date_start, date_end) &
                    self.date_in_timeframe(datetime(year=year, month=month, day=first_day),
                                           date_start, date_end)):
                continue
            cal = self.load_obj(file)
            for key in cal.keys():
                info = key.split("_")
                y = info[0][:4]
                m = info[0][4:6]
                d = info[0][6:]
                loc = info[1][:-1] if len(info[1]) == 4 else info[1]
                date = datetime(year=int(y), month=int(m), day=int(d))
                if not self.date_in_timeframe(date, date_start, date_end):
                    continue
                if location:
                    if loc == location.lower():
                        out.update({key: cal.get(key)})
                else:
                    out.update({key: cal.get(key)})
        return out

    def combine_data_sets(self, data_1: pd.DataFrame(), data_2: pd.DataFrame(), data_name: [],
                      n_jobs=1, batch_size=250, max_diff=86400):
        if not isinstance(data_name, type([])):
            data_name = [data_name]
        out_data = []
        data_1 = data_1.reset_index(drop=True)
        data_2 = data_2.reset_index(drop=True)
        self.pb = ProgressBar(total=len(data_1), prefix='', suffix='', decimals=3, length=50,
                              fill='=', zfill='>')
        self.pb.print_progress_bar(0)
        for year in data_1["YYYY"].unique():
            for month in data_1[data_1["YYYY"] == year]["MM"].unique():
                for day in data_1[(data_1["YYYY"] == year) & (data_1["MM"] == month)]["DD"].unique():
                    day_data_1 = data_1.loc[(data_1["YYYY"] == year) & (data_1["MM"] == month) &
                                            (data_1["DD"] == day)].copy().reset_index(drop=True)
                    day_data_2 = data_2.loc[(data_2["YYYY"] == year) & (data_2["MM"] == month) &
                                            (data_2["DD"] == day)].copy().reset_index(drop=True)
                    if (len(day_data_1) == 0) | (len(day_data_2) == 0):
                        for _ in range(len(day_data_1)):
                            self.pb.next()
                        continue
                    seconds = day_data_2["SS"]
                    values = day_data_2.loc[:, data_name]
                    datas = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer="threads")\
                                    (delayed(self.__collocate_data)(index, row, seconds, values,
                                                                    data_name, max_diff)
                                     for index, row in day_data_1.iterrows())
                    datas = [el for el in datas if el]
                    datas = pd.DataFrame.from_dict(datas)
                    if len(datas) == 0:
                        continue
                    datas = datas.set_index(datas["index"]).drop("index", axis=1)
                    day_data_1 = day_data_1.merge(datas, left_index=True, right_index=True)
                    out_data.append(day_data_1)
        del self.pb
        return pd.concat(out_data)

    def __collocate_data(self, data_1_index, data_1_row, seconds, values, data_name, max_diff):
        ss = data_1_row["SS"]
        diffs = np.abs(seconds - ss)
        i = np.argmin(diffs)
        diff = diffs[i]
        if diff > max_diff:
            self.pb.next()
            return
        value = values.loc[diffs.axes[0][i], :]
        value = np.array(value).flatten()
        self.pb.next()
        out_frame = {data_name[i]: value[i] for i in range(len(data_name))}
        out_frame.update({"index": data_1_index})
        out_frame.update({"diff": diff})
        return out_frame

    @staticmethod
    def __month_year_iter(start_year, start_month, end_year, end_month):
        ym_start = 12 * start_year + start_month - 1
        ym_end = 12 * end_year + end_month
        for ym in range(ym_start, ym_end):
            y, m = divmod(ym, 12)
            yield y, m + 1
