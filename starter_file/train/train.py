# +
import argparse
import os

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import azureml.core
from azureml.core import Workspace, Experiment, Run

import joblib


# -

def get_x_y(data):
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")
    
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--bootstrap', type=bool, default=False, help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
    parser.add_argument('--max_depth', type=int, default=-1, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument('--max_features', type=str, default="", help="The number of features to consider when looking for the best split.")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    
    args, leftovers = parser.parse_known_args() #parser.parse_args()

    if run.identity.startswith('OfflineRun'):
        interactive_run.log("bootstrap:", np.str("Yes") if np.bool(args.bootstrap) else np.str("No"))
        interactive_run.log("max_depth:", np.str("None") if np.int(args.max_depth) == -1 else np.int(args.max_depth))
        interactive_run.log("max_features:", np.str("None") if args.max_features == "" else args.max_features)
        interactive_run.log("min_samples_leaf:", np.int(args.min_samples_leaf))
        interactive_run.log("min_samples_split:", np.int(args.min_samples_split))
        interactive_run.log("n_estimators:", np.int(args.n_estimators))
    else:
        run.log("bootstrap:", np.str("Yes") if np.bool(args.bootstrap) else np.str("No"))
        run.log("max_depth:", np.str("None") if np.int(args.max_depth) == -1 else np.int(args.max_depth))
        run.log("max_features:", np.str("None") if args.max_features == "" else args.max_features)
        run.log("min_samples_leaf:", np.int(args.min_samples_leaf))
        run.log("min_samples_split:", np.int(args.min_samples_split))
        run.log("n_estimators:", np.int(args.n_estimators))

    model = RandomForestClassifier(bootstrap=args.bootstrap,
                                   max_depth=None if np.int(args.max_depth) == -1 else np.int(args.max_depth),
                                   max_features=None if args.max_features == "" else args.max_features,
                                   min_samples_leaf=np.int(args.min_samples_leaf),
                                   min_samples_split=np.int(args.min_samples_split),
                                   n_estimators=np.int(args.n_estimators),
                                   random_state=2653).fit(x_train, y_train)
    
    # Make predictions for the test set
    y_pred_test = model.predict(x_test)

    auc_weighted = roc_auc_score(y_test, y_pred_test, average="weighted")
    
    if run.identity.startswith('OfflineRun'):
        interactive_run.log("AUC_weighted", np.float(auc_weighted))
        interactive_run.complete()
    else:
        run.log("AUC_weighted", np.float(auc_weighted))
    
    # Save model as -pkl file to the outputs/ folder to use outside the script
    OUTPUT_DIR='./outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_file_name = 'heart_failure_hyperdrive.pkl'
    joblib.dump(value=model, filename=os.path.join(OUTPUT_DIR, model_file_name))

# Check library versions
print("SDK version:", azureml.core.VERSION)
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The joblib version is {}.'.format(joblib.__version__))
print('The pandas version is {}.'.format(pd.__version__))
#print('The sklearn_pandas version is {}.'.format(sklearn_pandas.__version__))

# +
from azureml.core import Dataset

run = Run.get_context()

if run.identity.startswith('OfflineRun'):
    ws = Workspace.from_config()
    
    experiment_name = 'heart-failure-clinical-data'
    experiment = Experiment(ws, experiment_name)
    
    interactive_run = experiment.start_logging()
else:
    ws = run.experiment.workspace


ds = Dataset.get_by_name(ws, name='Heart Failure Prediction')
# -

x, y = get_x_y(ds)

# +
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=223)
x_train.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
# -

if __name__ == '__main__':
    main()


