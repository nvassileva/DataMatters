# Usage Guide

First, some relevant road traffic data needs to be downloaded, for instance from Caltrans or directly using the links provided in the metadata.yml file stored in the data directory.

Then, data_gen_1BS.py can be used to generate mobile cellular traffic for the first base station in the sequence. Next, use data_gen_2BS.py to propagate the handover traffic down the sequence one base station at a time and to generate new call arrivals at a target base station.

After generating network statistics for all the base stations from the sequence, collect the relevant data from the generated XLSX files to a single csv file with the merging_excel_files.py for the first BS and merging_excel_sheets_files.py for all subsequent base stations.

Run the simulations with forecasting_lstm_tf.py.
