# End-milling-cutter-Lifetime-classifiication-ML-model

The life classification learning model uses the input quantitative feature report to predict the score weight of the results such as the tool life level through the model. 

The highest weight ratio is the final judgment result of the model.

Tool life classification Therefore, the input samples are all worn tools that are compared with the nominal size tool library (new tool tool library). 

Therefore, the new class tool option (New Class) appears in the middle of the classification model.

The results of the three classifications are code-named scrapped tools (Class = Broken), rough-cut tools (Class = Rough), and finishing tools (Class = Precise).

Split the data randomly at 7:3, and divide the feature data set into training data and test data. 

The training data will be randomly divided into ten groups of model validation data sets and model training in a 1:9 manner Data set, and use the K-Fold cross-evaluation method to evaluate the selected model of choice. 

During model training, the hyperparameters in the model are fine-tuned to minimize the loss function of the classification model, that is, to adjust the hyperparameter settings to optimize the performance of the model. 

This research uses a grid search algorithm , The adjustable hyperparameters are interactively combined in the model like a grid, and the training situations under all hyperparameter combinations are listed, and the best model performance is searched for in an exhaustive manner. 

After the model is trained and formed, the machine learning process needs to perform the final model performance test based on the characteristics of the test data set. 

The test set data that is split here are all randomly assigned and the model has not seen the feature set, so that the final model can be tested The classification accuracy achieved (accuracy).
