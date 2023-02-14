class Device:
    def __init__(self, device_name, device_db_key, on_threshold,
                 train_buildings, test_building, window_size, mean, std, memax):
        self.device_name = device_name
        self.device_db_key = device_db_key
        self.on_threshold = on_threshold
        self.train_buildings = train_buildings
        self.test_building = test_building
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self.memax = memax

    def get_save_path(self):
        return "experiments/" + self.device_db_key + "/"


def get_device_conf(device_key):
    switch = {
        "dish_washer": Device(device_name="dish washer", device_db_key=device_key, on_threshold=10,
                              train_buildings=[1, 2], test_building=5, window_size=100, mean=700, std=1000, memax=3000),
        "fridge": Device(device_name="fridge", device_db_key=device_key, on_threshold=50,
                         train_buildings=[1, 2, 4], test_building=5, window_size=50, mean=200, std=400, memax=200),
        "washing_machine": Device(device_name="washer dryer", device_db_key=device_key, on_threshold=20,
                                  train_buildings=[1, 5], test_building=2, window_size=50, mean=400, std=700,
                                  memax=2500),
        "microwave": Device(device_name="microwave", device_db_key=device_key, on_threshold=200,
                            train_buildings=[1, 2], test_building=5, window_size=50, mean=500, std=800,
                            memax=3000),
        "kettle": Device(device_name="kettle", device_db_key=device_key, on_threshold=1700,
                         train_buildings=[1, 2, 3, 4], test_building=5, window_size=100, mean=500, std=1000,
                         memax=3000),
    }
    return switch.get(device_key, "Invalid input")
