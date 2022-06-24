

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import argparse # 追加

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# enable auto logging
mlflow.autolog()

def main(args):

    with mlflow.start_run():
        # file_path = './data/daily-bike-share.csv'
        bike_data = pd.read_csv(args.input_data)
        # bike_data.head()

        # Try these hyperparameter values

        params = {
            'regressor__n_estimators': list(range(20, 101, 10)),
            'regressor__learning_rate': list(np.arange(0.05, 0.20, 0.01))
            }
        mlflow.log_params(params)

        X_train, X_test, y_train, y_test = data_prep(bike_data)
        model = train_model(X_train, y_train,params)
        metric ,fig  = evaluate_model(model,X_test,y_test)

        mlflow.log_figure(fig,"evaluate.png")
        mlflow.log_metrics(metric)
        mlflow.sklearn.log_model(model,artifact_path="model")
        
# Dataprep
def data_prep(bike_data):

    # Separate features and labels
    X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
    print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

    # Split data 70%-30% into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

    return X_train, X_test, y_train, y_test
    
# Train

def train_model(X_train, y_train,params):

    # Train the model


    # Define preprocessing for numeric columns (scale them)
    numeric_features = [6,7,8,9]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = [0,1,2,3,4,5]
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create preprocessing and training pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor())])

    score = make_scorer(r2_score)
    gridsearch = GridSearchCV(pipeline, params, scoring=score, cv=3, return_train_score=True)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_

    print("Best parameter combination:", gridsearch.best_params_, "\n")

    print (model)

    return model

# Evaluate
def evaluate_model(model,X_test,y_test):

    predictions = model.predict(X_test)
    np.set_printoptions(suppress=True)
    print('Predicted labels: ', np.round(predictions)[:10])
    print('Actual labels   : ' ,y_test[:10])
    
    fig = plt.figure(figsize=(10,10))
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions')
    # overlay the regression line
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
   
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    r2 = r2_score(y_test, predictions)
    print("R2:", r2)

    metric = {
        "MSE": mse,
        "RMSE":rmse,
        "R2":r2
    }
    
    return metric , fig
    
def parse_args():
   
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)

    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
