# Data Matters: The Case of Predicting Mobile Cellular Traffic


This is a public repository for the paper "Data matters: the case of predicting mobile cellular traffic".

## Description
The repository contains a data directory with all the necessary information to recreate the data with which the machine learning model is fed. Specifically, it contains the metadata of the original PeMS files used to generate mobile cellular network statistics and the input values (per base station) of the parameters of the code created for data generation.  The code directory contains: data_gen_1BS.py and data_gen_2BS.py, which are programs for generating mobile cellular network statistics, and  forecasting-lstm-tf.py, which is a machine learning program for training and testing the LSTM model used in the simulations. The env.yml file can be used to recreate the conda environment used for the simulations

## Requirements
Tensor Flow 2.17.0 (see the tf_env.yml file in the code directory)

## Results 
Simulation outcomes for additional data periods and splits for the mixture of two log-normally distributed call durations are available in the results directory. Simulation results for a exponentially distributed call duration setting are included too.

## Contribution
NV conceptualized and designed the study, developed the methodology and created the code for generating the data, created a basic version of a code for processing the results, ran the simulations, analyzed the results, wrote the paper and prepared the tables and figures.

MH wrote the code for running the simulations on the Triton computer cluster at Aalto University, wrote the code for collecting and summarizing the results and  contributed to the creation of the figures.

PI actively contributed to all of research phases: design of the study, analysis of the results and editing of the manuscript. 

## Acknowledgment 
The authors gratefully acknowledge the support received from Academy of Finland via the Centre of Excellence in Randomness and Structures, decision number 346308.

The calculations presented in the results directory were performed using computer resources within the Aalto University School of Science “Science-IT” project.



## License 
 <p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p> 
