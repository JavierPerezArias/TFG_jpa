import os
import sys
import re
import time
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.utils import shuffle
from sklearn.utils import parallel_backend

from sklearn.ensemble import IsolationForest

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

from pyspark.sql import SparkSession

# instantiate spark session
spark = SparkSession.builder.master("local[5]").appName("Test").getOrCreate()
sc = spark.sparkContext

from joblibspark import register_spark
register_spark()

data_dir = "./datos"

def get_data(data_dir: str) -> pd.DataFrame:
    
    data = pd.DataFrame()

    for path in os.listdir(data_dir):
        dir_content = os.path.join(data_dir, path)
        if os.path.isfile(dir_content):
            data = data.append(pd.read_csv(dir_content))

    return data

def process_data(data: pd.DataFrame):
    
    subnet1 = "192\.168\.8\.\d{1,3}"
    subnet2 = "192\.168\.3\.\d{1,3}"
    subnet3 = "200\.175\.2\.\d{1,3}"
    subnet4 = "192\.168\.20\.\d{1,3}"
    subnet5 = "172\.17\.\d{1,3}\.\d{1,3}"
    
    data["Label"].replace({"DDoS ": "DDoS"}, inplace=True)
    data.drop("Flow ID", inplace=True, axis=1)
    
    data["Src 192.168.8.0/24"] = data["Src IP"].str.match(subnet1)
    data["Src 192.168.3.0/24"] = data["Src IP"].str.match(subnet2)
    data["Src 200.175.2.0/24"] = data["Src IP"].str.match(subnet3)
    data["Src 192.168.20.0/24"] = data["Src IP"].str.match(subnet4)
    data["Src 172.17.0.0/16"] = data["Src IP"].str.match(subnet5)
    data["Src exterior ip"] = ~data["Src IP"].str.match("(" + subnet1 + "|" + subnet2 + "|" + subnet3 + "|" + subnet4 + "|" + subnet5 + ")")
    
    data["Src 192.168.8.0/24"] = data["Src 192.168.8.0/24"].astype(int)
    data["Src 192.168.3.0/24"] = data["Src 192.168.3.0/24"].astype(int)
    data["Src 200.175.2.0/24"] = data["Src 200.175.2.0/24"].astype(int)
    data["Src 192.168.20.0/24"] = data["Src 192.168.20.0/24"].astype(int)
    data["Src 172.17.0.0/16"] = data["Src 172.17.0.0/16"].astype(int)
    data["Src exterior ip"] = data["Src exterior ip"].astype(int)
    
    data["Dst 192.168.8.0/24"] = data["Dst IP"].str.match(subnet1)
    data["Dst 192.168.3.0/24"] = data["Dst IP"].str.match(subnet2)
    data["Dst 200.175.2.0/24"] = data["Dst IP"].str.match(subnet3)
    data["Dst 192.168.20.0/24"] = data["Dst IP"].str.match(subnet4)
    data["Dst 172.17.0.0/16"] = data["Dst IP"].str.match(subnet5)
    data["Dst exterior ip"] = ~data["Dst IP"].str.match("(" + subnet1 + "|" + subnet2 + "|" + subnet3 + "|" + subnet4 + "|" + subnet5 + ")")
    
    data["Dst 192.168.8.0/24"] = data["Dst 192.168.8.0/24"].astype(int)
    data["Dst 192.168.3.0/24"] = data["Dst 192.168.3.0/24"].astype(int)
    data["Dst 200.175.2.0/24"] = data["Dst 200.175.2.0/24"].astype(int)
    data["Dst 192.168.20.0/24"] = data["Dst 192.168.20.0/24"].astype(int)
    data["Dst 172.17.0.0/16"] = data["Dst 172.17.0.0/16"].astype(int)
    data["Dst exterior ip"] = data["Dst exterior ip"].astype(int)
    
    data[["Day", "Hour"]] = data["Timestamp"].str.split(" ", 1, expand=True)
    data[["Hour","Minute","PM"]] = data["Hour"].str.split(":", 2, expand=True)
    data[["Day","PM"]] = data["Day"].str.split("/", 1, expand=True)
    data["PM"] = data["Timestamp"].str.match(".*PM$")
    
    data["PM"] = data["PM"].astype(int)
    data["PM"] = 12 * data["PM"]
    data["Hour"] = data["Hour"].astype(int) + data["PM"]
    data["Minute"] = data["Minute"].astype(int)
    data["Day"] = data["Day"].astype(int)
    
    data["Hour sin"] = np.sin(data["Hour"]*(2.*np.pi/24))
    data["Hour cos"] = np.cos(data["Hour"]*(2.*np.pi/24))
    data["Minute sin"] = np.sin(data["Minute"]*(2.*np.pi/60))
    data["Minute cos"] = np.cos(data["Minute"]*(2.*np.pi/60))
    data["Day sin"] = np.sin((data["Day"]-1)*(2.*np.pi/31))
    data["Day cos"] = np.cos((data["Day"]-1)*(2.*np.pi/31))
    
    data["Attack"] = ~data["Label"].str.match("Normal")
    data["Attack"] = data["Attack"].astype(int)
    
    data.drop("Hour", inplace=True, axis=1)
    data.drop("Minute", inplace=True, axis=1)
    data.drop("Day", inplace=True, axis=1)
    data.drop("PM", inplace=True, axis=1)
    
    data.drop("Src IP", inplace=True, axis=1)
    data.drop("Dst IP", inplace=True, axis=1)
    data.drop("Timestamp", inplace=True, axis=1)
    data.drop("Label", inplace=True, axis=1)
    
    #columnas irrelevantes
    data.drop("Fwd PSH Flags", inplace=True, axis=1)
    data.drop("Fwd URG Flags", inplace=True, axis=1)
    data.drop("CWE Flag Count", inplace=True, axis=1)
    data.drop("ECE Flag Cnt", inplace=True, axis=1)
    data.drop("Fwd Byts/b Avg", inplace=True, axis=1)
    data.drop("Fwd Pkts/b Avg", inplace=True, axis=1)
    data.drop("Fwd Blk Rate Avg", inplace=True, axis=1)
    data.drop("Bwd Byts/b Avg", inplace=True, axis=1)
    data.drop("Bwd Pkts/b Avg", inplace=True, axis=1)
    data.drop("Bwd Blk Rate Avg", inplace=True, axis=1)
    data.drop("Init Fwd Win Byts", inplace=True, axis=1)
    data.drop("Fwd Seg Size Min", inplace=True, axis=1)
    
    data = shuffle(data)
    data.reset_index(drop=True, inplace=True)    
    
    return data
    
def isof(data: pd.DataFrame):
    
    y = np.array(data["Attack"])
    x = data.drop("Attack", axis = 1)
    
    isof = IsolationForest(bootstrap=True, contamination=0.0001, n_jobs=-1)
    y_isof = isof.fit_predict(x, y)
    
    x_isof = x.drop(x.index[np.asarray(np.where(y_isof == -1)).tolist()[0]])
    y_isof = np.delete(y, np.asarray(np.where(y_isof == -1)).tolist()[0])
    
    print("Ctcas:", x_isof.shape, x.shape)
    
    return x_isof,y_isof 

def rfe(x_isof: pd.DataFrame, y_isof: np.array, seed: int):
    
    rfe = RFECV(DecisionTreeClassifier(random_state=seed), step=1, n_jobs=-1)
    rfe.fit(x_isof, y_isof)
    
    x_isof_rfe = rfe.transform(x_isof)
    
    print("Ctcas:", x_isof_rfe.shape, x_isof.shape)
    
    return x_isof_rfe,y_isof

def rf(x_isof_rfe: pd.DataFrame, y_isof: np.array, seed: int):
    
    rf = RandomForestClassifier(random_state=seed)
    grid = {"n_estimators": [100, 50, 10], "max_depth": [None, 5], "max_features": ["sqrt"]}
    scorer = {"accuracy": "accuracy", "kappa": make_scorer(cohen_kappa_score),"f1": "f1","precision": "precision", "recall": "recall"}
    gscv = GridSearchCV(rf, grid, scoring=scorer, refit="accuracy", n_jobs=-1)
    
    with parallel_backend('spark', n_jobs=-1):
    	result = cross_validate(gscv, x_isof_rfe, y_isof, scoring=scorer, return_estimator=True, n_jobs=-1)
    
    for results in result["estimator"]:
        print(">>> Metrics (Mean It. - CV) <<<")
        print("Accuracy: ", results.cv_results_["mean_test_accuracy"])
        print("Kappa: ", results.cv_results_["mean_test_kappa"])
        print("F1: ", results.cv_results_["mean_test_f1"])
        print("Precision: ", results.cv_results_["mean_test_precision"])
        print("Recall: ", results.cv_results_["mean_test_recall"])
        print("Fit time: ", results.cv_results_["mean_fit_time"])
        print("Score time: ", results.cv_results_["mean_score_time"])
                
        print("Best score: ", results.best_score_)
        print("Best params: ", results.best_params_, "\n")
        
    print(">>> Metrics <<<")
    print("Accuracy: ", result["test_accuracy"])
    print("Kappa: ", result["test_kappa"])
    print("F1: ", result["test_f1"])
    print("Precision: ", result["test_precision"])
    print("Recall: ", result["test_recall"])
        
    return 

def svm(x_isof_rfe: pd.DataFrame, y_isof: np.array, seed: int):
    
    svm = LinearSVC(random_state=seed)
    grid = {"tol": [1e-3, 1e-4], "C": [100, 10, 1], "dual": [False]}
    scorer = {"accuracy": "accuracy", "kappa": make_scorer(cohen_kappa_score),"f1": "f1","precision": "precision", "recall": "recall"}
    gscv = GridSearchCV(svm, grid, scoring=scorer, refit="accuracy", n_jobs=-1)
    
    with parallel_backend('spark', n_jobs=-1):
    	result = cross_validate(gscv, x_isof_rfe, y_isof, scoring=scorer, return_estimator=True, n_jobs=-1)
    
    for results in result["estimator"]:
        print(">>> Metrics (Mean It. - CV) <<<")
        print("Accuracy: ", results.cv_results_["mean_test_accuracy"])
        print("Kappa: ", results.cv_results_["mean_test_kappa"])
        print("F1: ", results.cv_results_["mean_test_f1"])
        print("Precision: ", results.cv_results_["mean_test_precision"])
        print("Recall: ", results.cv_results_["mean_test_recall"])
        print("Fit time: ", results.cv_results_["mean_fit_time"])
        print("Score time: ", results.cv_results_["mean_score_time"])
                
        print("Best score: ", results.best_score_)
        print("Best params: ", results.best_params_, "\n")
        
    print(">>> Metrics <<<")
    print("Accuracy: ", result["test_accuracy"])
    print("Kappa: ", result["test_kappa"])
    print("F1: ", result["test_f1"])
    print("Precision: ", result["test_precision"])
    print("Recall: ", result["test_recall"])
        
    return 

seed = 1999
np.random.seed(seed)
data = process_data(get_data(data_dir))

t0 = time.time()
x, y = isof(data)
t1 = time.time()
print(f"ISOF: {t1-t0}\n")

t0 = time.time()
x, y = rfe(x, y, seed)
t1 = time.time()
print(f"RFE: {t1-t0}\n")

print("---RF---")
t0 = time.time()
rf(x, y, seed)
t1 = time.time()
print(f"RF: {t1-t0}\n")

print("---L.SVM---")
t0 = time.time()
svm(x, y, seed)
t1 = time.time()
print(f"L-SVC: {t1-t0}\n")
