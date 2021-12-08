#Homework 07

*Files:*

main.py : calls functions for data loading and preprocessing
load_data.py : Generator Function and Wrapper Generator, Splitting function and function for creating+splitting ds
data_pipeline.py : Datapipeline


*To-Do:*

- yield comments for integration_task and my_integration_task (unter function definition)

- Build LSTM model:
    - LSTM_cell class
    - LSTM_layer class
    - final wrapper model class

- Unrolling of LSTM (in graph mode?)

- Training (>80% accuracy)


*Notes:*

- !!!if seq_len ändern (in integration_task) auch output_signature im from_generator ändern


*Questions:*

- Which steps do we need in the data pipeline? (input normalization doesn't make sense because input ist sowieso schon normalverteilt?)
