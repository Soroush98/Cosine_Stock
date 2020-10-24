# Stock
## Stock and bitcoin prediction with an innovative method

1-a window size is considered. 10 top similar period of time when movements are the same as current window size will be chosed from historical data.\
2-Data is labeled whether is going up or down in next week. 1 for going up and 0 for going down\
3-Training will be done by mlp or lstm for bitcoin or stock markets.\
4-Now Prediction for next week can be done by feeding last window to trained network.

Results shows almost 70% validation accuracy
** Note that data is too little therefore algorithm may not be good enough. 
