#!/bin/bash


gnome-terminal -- bash -c "python server.py test.csv 30; exec bash"

sleep 5

gnome-terminal -- bash -c "python client.py client1.csv prevalentHyp; exec bash"

gnome-terminal -- bash -c "python client.py client2.csv TenYearCHD; exec bash"

echo "All scripts are running in parallel."
