# Traffic_Data_Projection_Using_GCN
Using GCN and ChebNet implement traffic flow data prediction.

The forecasting task of traffic flow in the city is a typical time series forecasting task. However, the difference from traditional time series analysis is that the city's sites present a graph structure. Therefore, the project uses several common graph convolution models to achieve the task of node traffic prediction. 

The data is from the public data set, I have prepared it for everyone, under the PeMS_04 folder.

In data_processing.py, we give the processing method for time series such as traffic flow. 
In model.py, we give the implementation of three graph convolutional networks, namely GCN, ChebNet, and GAT. 
You can train and predict through train.py.
