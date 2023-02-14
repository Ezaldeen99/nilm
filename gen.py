""" This script generates train sets from several building data"""
from __future__ import print_function, division

import os
import shutil
import sys
import urllib

import numpy as np
from nilmtk import DataSet
from nilmtk.electric import align_two_meters

from utils.device_class import get_device_conf

ukdale_windows = [
    ("2014-9-1", "2014-9-30"),
    ("2013-9-1", "2013-9-30"),
    ("2013-3-1", "2013-3-30"),
    ("2013-4-7", "2013-5-7"),
    ("2014-9-1", "2014-9-30"),
]
key_names = ['fridge', 'microwave', 'dish_washer', 'kettle', 'washing_machine']

train_size = 200000


def create_trainset(meter, mains, train_size, window_size):
    """Creates a time series from the raw UKDALE DataSet """
    all_x_train = np.empty((train_size, window_size, 1))
    all_y_train = np.empty((train_size,))
    low_index = 0

    gen = align_two_meters(meter, mains)
    for chunk in gen:
        if chunk.shape[0] < 3000:
            continue
        chunk.fillna(method='ffill', inplace=True)
        x_batch, y_batch = generate_batch(chunk.iloc[:, 1], chunk.iloc[:, 0], chunk.shape[0] - window_size, 0,
                                          window_size)
        high_index = min(len(x_batch), train_size - low_index)
        all_x_train[low_index:high_index + low_index] = x_batch[:high_index]
        all_y_train[low_index:high_index + low_index] = y_batch[:high_index]
        low_index = high_index + low_index
        if low_index == train_size:
            break

    return all_x_train, all_y_train


def generate_batch(mainchunk, meterchunk, batch_size, index, window_size):
    """Generates batches from dataset

    Parameters
    ----------
    index : the index of the batch
    """
    offset = index * batch_size
    x_batch_list = []
    for i in range(batch_size):
        x_batch_list.append(mainchunk[i + offset:i + offset + window_size])
    x_batch = np.array(x_batch_list)

    y_batch = meterchunk[window_size - 1 + offset:window_size - 1 + offset + batch_size]
    x_batch = np.reshape(x_batch, (len(x_batch), window_size, 1))

    return x_batch, y_batch


def get_test_data(building_number, meter_key):
    """Opens dataset of synthetic data from Neural NILM

    Parameters
    ----------
    building_number : The id of the building
    meter_key : The db key of the meter
    """

    path = "dataset/ground_truth_and_mains/"
    main_filename = "{}building_{}_mains.csv".format(path, building_number)
    meter_filename = "{}building_{}_{}.csv".format(path, building_number, meter_key)
    mains_reading = np.genfromtxt(main_filename)
    meter_reading = np.genfromtxt(meter_filename)
    if len(mains_reading) != len(meter_reading):
        min_length = min(len(meter_reading), len(meter_reading))
        mains_reading = mains_reading[:min_length]
        meter_reading = meter_reading[:min_length]

    return mains_reading, meter_reading


def download_dataset():
    print("Downloading test dataset for the first time")
    os.makedirs("dataset")
    urllib.request.urlretrieve("http://jack-kelly.com/files/neuralnilm/NeuralNILM_data.zip", "dataset/ds.zip")
    import zipfile

    zip_ref = zipfile.ZipFile('dataset/ds.zip', 'r')
    zip_ref.extractall('dataset')
    zip_ref.close()
    os.remove("dataset/ds.zip")
    shutil.rmtree("dataset/disag_estimates", ignore_errors=True)
    os.makedirs("dataset/trainsets")
    print("Done downloading")


def generate_dataset(dateset_path):
    # get the test dataset if it is not there
    if not os.path.exists("dataset"):
        download_dataset()

    dataset = DataSet(dateset_path)
    for device in key_names:
        device_config = get_device_conf(device)
        try:
            os.makedirs(device_config.get_save_path())
        except Exception as e:
            print(e)

        # Create trainset for meter
        print(device_config.device_name)
        house_keys = device_config.train_buildings
        window_size = device_config.window_size
        all_x_train = np.empty((train_size * len(house_keys), window_size, 1))
        all_y_train = np.empty(train_size * len(house_keys))
        for index, building in enumerate(house_keys):
            dataset.set_window(start=(ukdale_windows[building - 1])[0], end=(ukdale_windows[building - 1])[1])
            electric_meter = dataset.buildings[building].elec
            meter = electric_meter[device_config.device_name]
            mains = electric_meter.mains()
            all_x, all_y = create_trainset(meter, mains, train_size, window_size)
            start_index = index * train_size
            end_index = (index + 1) * train_size
            all_x_train[start_index:end_index] = all_x
            all_y_train[start_index:end_index] = all_y

        np.save('dataset/trainsets/X-{}'.format(device_config.device_db_key), all_x_train)
        np.save('dataset/trainsets/Y-{}'.format(device_config.device_db_key), all_y_train)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen.py ukdale_path")
        exit()

    generate_dataset(sys.argv[1])
