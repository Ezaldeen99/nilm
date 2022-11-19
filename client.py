from __future__ import print_function, division

import json

import flwr as fl
import numpy as np

from gen import opends
from model import create_model

allowed_key_names = ['fridge', 'microwave', 'dish_washer', 'kettle', 'washing_machine']
key_name = "dish_washer"


def normalize(data, mmax):
    return data / mmax


# =======  Open configuration file
if key_name not in allowed_key_names:
    print("    Device {} not available".format(key_name))
    print("    Available device names: {}", allowed_key_names)
conf_filename = "appconf/{}.json".format(key_name)
with open(conf_filename) as data_file:
    conf = json.load(data_file)

input_window = conf['lookback']
threshold = conf['on_threshold']
mamax = 5000
memax = conf['memax']
mean = conf['mean']
std = conf['std']
train_buildings = conf['train_buildings']
test_building = conf['test_building']
on_threshold = conf['on_threshold']
meter_key = conf['nilmtk_key']
save_path = conf['save_path']

# ======= Training phase
print("Training for device: {}".format(key_name))
print("    train_buildings: {}".format(train_buildings))

# Open train sets
x_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
print("X_train ", x_train)
x_train = normalize(x_train, mamax)
y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
print("y_train ", y_train)
y_train = normalize(y_train, memax)
model = create_model(input_window)

# ======= Disaggregation phase
mains, meter = opends(test_building, key_name)
x_test = normalize(mains, mamax)
y_test = meter


class NilmClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=7, batch_size=128)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=NilmClient())
