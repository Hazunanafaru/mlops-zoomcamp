import os
import pandas as pd
import datetime
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from prefect.task_runners import SequentialTaskRunner
from datetime import timedelta

@task
def get_paths(date):
    logging = get_run_logger()

    path = []
    parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    year = parsed_date.year
    months = (parsed_date.month-1, parsed_date.month-2)

    for month in months:
        if month > 10:
            logging.info(f"Will use fhv_tripdata_{year}-{month}.parquet dataset")
            path.append(f"../data/fhv_tripdata_{year}-{month}.parquet")
        else:
            logging.info(f"Will use fhv_tripdata_{year}-0{month}.parquet dataset")
            path.append(f"../data/fhv_tripdata_{year}-0{month}.parquet")
    
    # Return train_path and val_path
    return path[0], path[1]

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        # print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        # print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    # print(f"The shape of X_train is {X_train.shape}")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    # print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    # print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def export(date, dv, lr):
    logger = get_run_logger()

    # Export dv and model
    os.makedirs(f"models/{date}/",exist_ok = True)
    with open(f"models/{date}/dv-{date}.b", "wb") as dv_hand:
        pickle.dump(dv, dv_hand)
        logger.info(f"DictVectorizer saved in models/{date}/dv-{date}.b")

    with open(f"models/{date}/model-{date}.bin", "wb") as lr_hand:
        pickle.dump(lr, lr_hand)
        logger.info(f"Model saved in models/{date}/model-{date}.bin")

# @flow(task_runner=SequentialTaskRunner()) # for squential task runner
@flow
def main(date="2021-08-15"):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    export(date, dv, lr)

# main()

DeploymentSpec(
    flow=main,
    name="model_training",
    # schedule=IntervalSchedule(interval=timedelta(weeks=1)),
    schedule=CronSchedule(cron="0 9 15 * *", timezone="Asia/Jakarta"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml", "linear_reg"],
)