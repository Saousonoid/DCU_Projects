#!/bin/bash


gnome-terminal -- bash -c "python server.py test.csv 30; exec bash"

sleep 5

gnome-terminal -- bash -c "python client.py diabetes.csv diabetes 0 ; exec bash"

gnome-terminal -- bash -c "python client.py ckd.csv ckd 1; exec bash"

echo "All scripts are running in parallel."
