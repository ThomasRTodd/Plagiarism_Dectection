from __future__ import print_function

import argparse
import os
import pandas as pd

#from sklearn.externals import joblib
import imp
import joblib
## TODO: Import any additional libraries you need to define a model
import io
import numpy as np 
import boto3

#!pip install sagemaker

#from sagemaker import sagemaker.LinearLearner
from sklearn.linear_model import LogisticRegression

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    print(train_data.head)
    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    # convert features/labels to numpy
    train_x_np = train_x.astype('float32')
    train_y_np = train_y.astype('float32')

    # create RecordSet
   

    # TODO: Define a model 
    #model = LinearLearner(
    #                   train_instance_count=1, 
    #                   train_instance_type='ml.c4.2xlarge',
    #                   predictor_type='binary_classifier',
    #                   epochs=15)

    #formatted_train_data = model.record_set(train_x_np, labels=train_y_np)
    
    ## TODO: Train the model
    
    model = LogisticRegression(penalty='l1',solver='liblinear')
    model.fit(train_x,train_y)
    
    #model.fit(formatted_train_data)
    #model = LogisticRegression.fit(train_x_np, train_y_np)
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))