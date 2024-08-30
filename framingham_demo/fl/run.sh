#!/bin/bash


gnome-terminal -- bash -c "python server.py test.csv 30; exec bash"

sleep 5

gnome-terminal -- bash -c "python client.py client1.csv 0 prevalentHyp; exec bash"

gnome-terminal -- bash -c "python client.py client2.csv 1 TenYearCHD; exec bash"

echo "All scripts are running in parallel."
