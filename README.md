# NILM using Flower framework

A modified version of the Sequence-to-point network learning
to build an NILM using Flower framework to train multi clients


## Before Setup:

- Download ukDALE dataset from [here](http://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip)
- unzip the dataset to `data` folder

## To set up the project
Run
```bash
python gen.py <path to your UKDALE h5>
```
This will create the trainsets and download the Neural NILM test set. The trainset comes from the data used in Neural NILM. This may take some time.

Running the client step below will also try to run generate the dataset with default dataset path at `data\ukdale.h5`, if the user didn't run this method first
## To train and test the network
Run
```bash
python client.py <device>
```
Where device can be
* ```dishwasher```
* ```fridge```
* ```kettle```
* ```microwave```
* ```washing_machine```

Then Run

```bash
python server.py
```