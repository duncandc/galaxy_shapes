#!/bin/bash

python estimate_disk_fraction.py
python measure_shape_distributions.py ISO 1000
python measure_shape_distributions.py DEV 1000
python measure_shape_distributions.py EXP 1000
