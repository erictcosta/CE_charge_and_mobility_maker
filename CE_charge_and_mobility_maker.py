import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
import os
import numpy as np

#######   Inform the path ################
C4D_DATA = "C4D_data"
D_FILES = "d_files"
AGILENT_DATA = "experiments_data.txt"
##########################################

############  CE parameters ##############
LD1 = 13  # distance of detector 1 in cm
LD2 = 24  # distance of detector 2 in cm
LDUV = 29  # distance of UV detector in cm
LT = 50  # length of capillary
I = 23.32  # Current applied on the BGE
V = 24.016  # voltage applied on the BGE
###########################################


CE_DATA = "CE_data"
PROCESSSED_DATA = "processed_data" + os.sep

FILE_NAMES = "" 
PATH = "."

pd.set_option('display.max_columns', None)

def save_par(dic):
    with open("pareamentos.log", "w") as f:
        for key1, value1 in dic.items():
            f.write(f"{key1}  n = {len(value1)}\n")
            for key2, value2 in value1.items():
                f.write(f"{' ' * 4} {key2} : {value2}\n")


def save_data_to_file(data, file_name, order):
    data = data[order]
    data = data.fillna(0)
    data.to_csv(file_name, sep="\t", index=False)


def create_charge_column(data, time_col, current_col):
    data_charge = data[time_col].diff() * data[current_col] * 60
    data["Charge_uC"] = data_charge.cumsum()
    return data


def create_mobility_column(data, q, LD, LT, I, V):
    k = LD * LT * I * 1E-6 / V / 1E3 / 1E-6
    data["Mobility"] = k / data[q]
    return data


def my_interpolator(keep_data, interpole_data, cols=("Time_m", "Signal")):
    commom_col = cols[0]
    detector_col = cols[1]
    df = pd.concat([keep_data, interpole_data])
    df.sort_values(by=[commom_col], inplace=True)
    df = df.groupby(commom_col).mean()
    df_detectors = df.iloc[:, :keep_data.shape[1]]
    df = df.interpolate(method="index")
    df.reset_index(inplace=True)
    df_detectors.reset_index(inplace=True)
    return df[~df_detectors[detector_col].isna()]

def make_interpole(df_detector, df, detector, key, LD1, LT, I, V):
    df_detector1 = my_interpolator(df_detector, df.iloc[:, 1:], ("Time_m", detector))
    df_detector1 = create_charge_column(df_detector1, "Time_m", "Current_uA")
    df_detector1 = create_mobility_column(df_detector1, "Charge_uC", LD1, LT, I, V)
    orders = ["Time_m", "Charge_uC", "Mobility", detector, "Current_uA", "Voltage_kV", "Power_w", "Tray_oC",
              "Cassette_oC", "LeakCurrent_uA", "lamp_current"]
    save_data_to_file(df_detector1, PROCESSSED_DATA + key + detector + ".inter.dat", orders)

def merger_and_interpole_files(dic):
    # LD1 = 13    # distance of detector 1 in cm
    # LD2 = 24    # distance of detector 2 in cm
    # LDUV = 29   # distance of UV detector in cm
    # LT = 50     # length of capillary
    # I = 23.32   # Current applied on the BGE
    # V = 24.016  # voltage applied on the BGE
    keys_names = ["CE1_B", "CEDAD1_B", "CEDAD1_C", "CEDAD1_D", "CE1_A", "CE1_C", "CE1_J", "CE1_K", "CE1_D", "CEDAD1_I"]
    titles = ["Current_uA", "abs_b", "abs_c", "abs_c", "Voltage_kV", "Power_w", "Tray_oC", "Cassette_oC", "LeakCurrent_uA", "lamp_current"]
    try:
        os.mkdir(PROCESSSED_DATA)
    except OSError as error:
        print(error)

    for key1, value1 in dic.items():
        c4d = pd.read_csv(value1["C4D"], sep="\t", names=["Time_m", "Detector1", "Detector2"])
        cedad = pd.read_csv(value1["CEDAD1_A"], sep="\t", names=["point", "Time_m", "abs_a"], header=1)
        del cedad["point"]
        df = pd.Series(dtype="float64")
        for key2, value2 in value1.items():
            if key2 not in ["C4D", "CEDAD1_A"]:
                signal_title = [j for i, j in zip(keys_names, titles) if i == key2]
                signal = pd.read_csv(value2, sep="\t", names=["point", "Time_m", signal_title[0]], header=1)
                del signal["point"]
                df = pd.concat([df, signal], ignore_index=True)

        make_interpole(c4d, df, "Detector1", key1, LD1, LT, I, V)
        make_interpole(c4d, df, "Detector1", key1, LD2, LT, I, V)
        make_interpole(cedad, df, "abs_a", key1, LDUV, LT, I, V)


def print_dict(dic):
     for key1, value1 in dic.items():
        print(key1, len(value1))
        for key2, value2 in value1.items():
            print(f"{' ' * 4} {key2} : {value2} ")


def get_intrument_data(file_name, experiments):
    files = {i: {} for i in experiments}
    for i in experiments:
        for j in file_name:
            if i + ".dat" in j:
                name = j.split(os.sep)[-1].split("_")[:2]
                files[i]["_".join(name)] = j

    return files


def get_modification_times(files_names):
    if isinstance(files_names, list):
        return [(os.stat(i).st_mtime, i) for i in files_names]
    else:
        return [(os.stat(files_names).st_mtime, files_names)]


def check_creatio_time_vs_datetime_on_file_name(data_file):
    date_in_file_name = [i[1].split("uM")[1][:-4] for i in data_file]
    times_in_seconds = [datetime.strptime(date, "%Y.%m.%d_%Hh%Mm%Ss").timestamp() for date in date_in_file_name]
    data_file_sliced = [i[1] for i in data_file]
    dates_in_file = [(i, j) for i, j in zip(times_in_seconds, data_file_sliced)]
    dates_in_file.sort()
    data_file.sort()
    return not (False in [True for i, j in zip(data_file, dates_in_file) if i[1] != j[1]])


def paring_files():
    data_ce = get_files_names(path=CE_DATA, ext="*/*.dat", recursive_=True)
    data_c4d = get_modification_times(get_files_names(path=C4D_DATA, ext="*.ele"))
    parameters_ce = get_modification_times(get_files_names(path=D_FILES, ext="*.d"))
    parameters_ce.sort()
    data_c4d.sort()

    if len(parameters_ce) != len(data_c4d) and not check_creatio_time_vs_datetime_on_file_name(data_c4d):
        print(f"alert! got {len(data_c4d)} files from C4D and {len(parameters_ce)} from Agilent")
        return None

    experiments = [i[1].split(os.sep)[-1][:-2] for i in parameters_ce]
    instrument_data = get_intrument_data(data_ce, experiments)
    pared = [(par1[1], par2[1]) for par1, par2 in zip(parameters_ce, data_c4d)]
    keys = [i for i in instrument_data]
    for i in keys:
        for j, z in pared:
            if i in j:
                instrument_data[i]["C4D"] = z

    return instrument_data


def simple_plot(file_name, data):
    plt.cla()
    plt.clf()
    plt.figure(1)
    plt.style.use('seaborn')
    for i in range(0, data.shape[1], 2):
        plt.plot(data.iloc[:, i], data.iloc[:,i+1])
    plt.ylabel(data.columns[1])
    plt.title(data.columns[0])
    name = f"{str(file_name[:-4])}_{data.columns[1]}_{data.columns[0]}.png"
    print(name)
    plt.savefig(name)

    return file_name


def get_files_names(file_names=None, path=".", ext="*", recursive_=True):
    if len(FILE_NAMES) <= 0:
        file_names = glob.glob(os.path.join(path, ext), recursive=recursive_)
    return file_names


def get_datas(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    return lines


def extract_agilent_files(data):
    file_openned = False
    count_CE = 0
    parameters_ce = get_files_names(file_names=[""], path="./d_files/", ext="*.d")
    experiments = [i.split(os.sep)[-1][:-2] for i in parameters_ce]
    pass_next = False
    try:
        os.mkdir(CE_DATA)
    except OSError as error:
        print(error)
    folder = CE_DATA
    for line in data:
        if pass_next:
            pass_next = False
            continue
        if "***NO DATA POINTS***" in line or "***ZERO ABUNDANCE***" in line:
            pass_next = True
            continue
        if '#"CE' in line:
            count_CE += 1
            if file_openned:
               f.close()
            f_name = line[2:]
            for i in ': ':
                f_name = f_name.replace(i, "_")
            f_name = f_name.replace('.d"\n', ".dat")
            f_name = f_name.replace('_-_', "_")
            for i in experiments:
                if i in f_name:
                    folder = CE_DATA + os.sep + i.split("-")[:-1][0]
                    try:
                        os.mkdir(folder)
                    except OSError as error:
                        pass
            f = open(os.path.join(folder, f_name), 'w')
            file_openned = True
        else:
            if file_openned and line != "":
                f.write(line)


def plot_inters_data():
    file_names = get_files_names(path=PROCESSSED_DATA, ext="*inter.dat", recursive_=False)
    datas = [pd.read_csv(i, sep="\t") for i in file_names]
    for i, j in zip(file_names, datas):
        for k in ["Detector1", "Detector2", "abs_a"]:
            if k in j.columns:
                simple_plot(i, j[["Time_m", k]])
                simple_plot(i, j[["Charge_uC", k]])
                mob = j[["Mobility", k]]
                mob = mob[mob["Mobility"] < 0.004]
                mob = mob[mob["Mobility"] > 0.000005]
                simple_plot(i, mob)


def extract_agilent_data():
    file_names = get_files_names(file_names=[AGILENT_DATA], ext="*txt")
    print(f"{file_names=}")
    for file_name in file_names:
        data = get_datas(file_name)
        print(f"{len(data)=}")
        extract_agilent_files(data)


if __name__ == '__main__':
    extract_agilent_data()
    datas_pared = paring_files()
    print_dict(datas_pared)
    save_par(datas_pared)
    merger_and_interpole_files(datas_pared)
    plot_inters_data()

