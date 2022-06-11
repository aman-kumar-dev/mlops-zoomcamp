from numpy import cumprod
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from xgboost import train
import datetime, os, pickle
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

@task
def read_data(path):
    logger = get_run_logger()
    logger.info(f'Reading parquet file: {path}')
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
        logger.info(f"The mean duration of training is {mean_duration}")
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
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

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    """ date in format YYYY-MM-DD"""
    curr_date = datetime.date.fromisoformat(date) if date != None else datetime.date.today()
    train_date = curr_date - datetime.timedelta(days=60)
    val_date = curr_date - datetime.timedelta(days=30)
    train_date = train_date.isoformat()
    val_date = val_date.isoformat()
    train_date = train_date[:-3]
    val_date = val_date[:-3]

    train_str = os.path.join('data', f'fhv_tripdata_{train_date}.parquet')
    val_str = os.path.join('data', f'fhv_tripdata_{val_date}.parquet')
    return train_str, val_str

@flow(task_runner=SequentialTaskRunner())
def main(date: str = None):

    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # Save model as "model-{date}.pkl"
    with open(f"models/model-{date}.pkl", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open(f"models/dv-{date}.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

# DeploymentSpec(
#     flow=main,
#     name="model_training",
#     schedule=IntervalSchedule(interval=datetime.timedelta(minutes=5)),
#     flow_runner=SubprocessFlowRunner(),
#     tags=["ml"]
# )
DeploymentSpec(
    flow=main,
    name="cron-schedule-deployment-homework",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["homework"]
)

if __name__ == '__main__':
    main(date="2021-08-15")