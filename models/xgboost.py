import xgboost
from sklearn.model_selection import train_test_split


def xgb_batch_train(x_train, y_train, batches, save_fname):
    '''
    This function trains xgboost models in batches in order to fit in the GPU's memory.
    The files are only loaded in memory once they are needed for training.
    Arguments:
    --  x_train :   A list of filepaths to TIF files to use for training
    --  y_train :   A list of filepaths to TIF files to use for labels
    '''


