from __future__ import print_function, division

import flwr as fl
import numpy as np

import metrics
from gen import get_test_data, generate_batch
from model import create_model
from utils.device_class import get_device_conf

allowed_key_names = ['fridge', 'microwave', 'dish_washer', 'kettle', 'washing_machine']
key_name = "washing_machine"


def normalize(data, mmax):
    return data / mmax


def denormalize(data, mmax):
    return data * mmax


# =======  Open configuration file
if key_name not in allowed_key_names:
    print("    Device {} not available".format(key_name))
    print("    Available device names: {}", allowed_key_names)
device_conf = get_device_conf(key_name)

input_window = device_conf.window_size
threshold = device_conf.on_threshold
epochs = 1
mamax = 5000
memax = device_conf.memax
mean = device_conf.mean
std = device_conf.std
train_buildings = device_conf.train_buildings
test_building = device_conf.test_building
meter_key = device_conf.device_db_key
save_path = device_conf.get_save_path()

# ======= Training phase
print("Training for device: {}".format(key_name))
print("    train_buildings: {}".format(train_buildings))

# Open train sets
x_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
x_train = normalize(x_train, mamax)
y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
y_train = normalize(y_train, memax)
model = create_model(input_window)
model.summary(line_length=100)

# ======= Test data ======== #
mains, meter = get_test_data(test_building, key_name)
x_test = normalize(mains, mamax)
y_test = meter


class NilmClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=epochs, batch_size=128)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        x_batch, y_batch = generate_batch(x_test, y_test, len(x_test) - input_window, 0, input_window)
        print("X_test ", x_test)
        print("y_test ", y_test)
        pred = model.predict(x_batch)
        print("pred ", pred)
        pred = denormalize(pred, memax)
        pred[pred < 0] = 0
        pred = np.transpose(pred)[0]
        # Save results
        np.save("{}pred-{}-epochs{}".format(save_path, key_name, epochs), pred)

        rpaf = metrics.recall_precision_accuracy_f1(pred, y_batch, threshold)
        rete = metrics.relative_error_total_energy(pred, y_batch)
        mae = metrics.mean_absolute_error(pred, y_batch)
        loss = metrics.mean_square_error(pred, y_batch)

        print("============ Recall: {}".format(rpaf[0]))
        print("============ Precision: {}".format(rpaf[1]))
        print("============ Accuracy: {}".format(rpaf[2]))
        print("============ F1 Score: {}".format(rpaf[3]))

        print("============ Relative error in total energy: {}".format(rete))
        print("============ Mean absolute error(in Watts): {}".format(mae))
        print("============ Loss: {}".format(loss))
        return loss, len(x_test), {"accuracy": rpaf[2]}


# Start Flower client
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=NilmClient())
