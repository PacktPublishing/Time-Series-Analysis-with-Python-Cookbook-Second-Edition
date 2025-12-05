import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sklearn.pipeline import make_pipeline as sklearn_make_pipeline
from sklearn.multioutput import MultiOutputRegressor

def handle_missing_data(df):
    n = int(df.isna().sum().sum()) # total missing values
    if n > 0:
        print(f'found {n} missing observations...')
        df.ffill(inplace=True)

def generate_lagged_features(df, window):
    """Transform time series data into a supervised learning format using sliding windows.
    
    Args:
        df (pd.DataFrame): Univariate time series data
        window (int): Number of time steps to use as features
        
    Returns:
        pd.DataFrame: DataFrame with lagged features (x_1 to x_n) and target (y)
    """
    # Validate input
    if not isinstance(window, int) or window < 1:
        raise ValueError("Window size must be a positive integer")
    
    # Convert DataFrame to 1D array
    d = df.values.squeeze()
    
    # Create sliding windows of past observations
    x = np.lib.stride_tricks.sliding_window_view(d, window_shape=window)[:-1]
    
    # Create target variable (next time step)
    y = d[window:]
    
    # Create column names for lagged features
    cols = [f'x_{i}' for i in range(1, window+1)]

    # Create DataFrames for features and target
    df_xs = pd.DataFrame(x, columns=cols, index=df.index[window:])
    df_y = pd.DataFrame(y, columns=['y'], index=df.index[window:])

    # Combine features and target into a single DataFrame
    return pd.concat([df_xs, df_y], axis=1)



def multiple_output_features(df, window_in, window_out):
    d = df.values.squeeze()

    x = np.lib.stride_tricks.sliding_window_view(d, window_shape=window_in)[:-window_out]
    y = np.lib.stride_tricks.sliding_window_view(d[window_in:], window_shape=window_out)

    cols_x = [f'x_{i}' for i in range(1, window_in + 1)]
    cols_y = [f'y_{i}' for i in range(1, window_out + 1)]
    
    df_xs = pd.DataFrame(x, columns=cols_x, index=df.index[window_in:len(df) - window_out + 1])
    df_ys = pd.DataFrame(y, columns=cols_y, index=df.index[window_in:len(df) - window_out + 1])

    return pd.concat([df_xs, df_ys], axis=1)

def split_data(df, test_split=0.10):
    """Split time series data into training and test sets.
    
    Args:
        df (pd.DataFrame): Time series data to split
        test_split (float, default=0.10): Proportion of data to use for testing
        
    Returns:
        tuple: (train_df, test_df)
    """
    n = int(len(df) * test_split)
    train, test = df[:-n], df[-n:]
    return train, test

class Standardize:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def _transform(self, df):
        return (df - self.mu) / self.sigma

    def fit_transform(self, train, test):
        # Calculate mean and std on training data only
        self.mu = train.mean()
        self.sigma = train.std()
        
        # Standardize training and test sets
        train_s = self._transform(train)
        test_s = self._transform(test)
        return train_s, test_s

    def transform(self, df):
        # Apply transformation to any new data
        return self._transform(df)

    def inverse(self, df):
        # Inverse transformation for the whole DataFrame
        return (df * self.sigma) + self.mu

    def inverse_y(self, df):
        # Inverse transformation specifically for the first column (y)
        return (df * self.sigma.iloc[-1]) + self.mu.iloc[-1]

def preprocess(df, split, window=5, generate_features=True):
    """
    Preprocess time series data by handling missing values, generating lagged features (optional),
    splitting into train/test sets, and standardizing.

    Args:
        df (pd.DataFrame): Input dataframe. Can be raw time series or already lagged features.
        split (float): Fraction of data to use for testing.
        window (int, optional): Window size for lagged features. Defaults to 5.
        generate_features (bool, optional): Whether to generate lagged features. 
                                            Set to False if df already contains features. Defaults to True.

    Returns:
        tuple: (train_scaled, test_scaled, scaler_object)
    """
    # Handle missing data
    handle_missing_data(df)
    
    if generate_features:
        # Generate lagged features
        df_os = _generate_lagged_features(df, window)
    else:
        df_os = df
    
    # Split data
    train, test = split_data(df_os, split)
    
    # Standardize data
    sc = Standardize()
    train_s, test_s = sc.fit_transform(train, test)
    
    return train_s, test_s, sc



def train_different_models_scaled(train, test, regressors, sc, train_func):
    results = []
    for reg_name, regressor in regressors.items():
        reg = regressor()
        results.append(train_func(train,
                                   test,
                                   reg,
                                   reg_name, sc))
    return results

def evaluate_results(results, by='MASE'):
    cols = ['Model Name', 'RMSE','MAPE', 'MASE']
    df_sorted = results[cols].sort_values(by).reset_index(drop=True)
    return df_sorted

def plot_results(results, data_name):
    cols = ['yhat', 'actual', 'Model Name']
    for row in results[cols].iterrows():
        yhat, actual, name = row[1]
        plt.title(f'{data_name} - {name}')
        plt.plot(actual, 'k--', alpha=0.5)
        plt.plot(yhat, 'k')
        plt.legend(['actual', 'forecast'])
        plt.show()


def train_different_models_mo(train, test, regressors, win_in, win_out, train_func):
    results = []
    for reg_name, regressor in regressors.items():
        result = train_func(
            train=train, 
            test=test, 
            regressor=regressor, 
            reg_name=reg_name, 
            win_in=win_in, 
            win_out=win_out
        )
        results.append(result)
    return results






def train_different_models_grid_cv(train, test, regressors, train_func):
    results = []
    for reg_name, (reg, param_grid) in regressors.items():
        results.append(train_func(train,
                                   test,
                                   reg(),
                                   reg_name,
                                   param_grid))
    return results

def get_best_params(df):
    best_params = df['Model'].best_params_
    return best_params

def contains_holiday(year, month):
    # US holidays
    # New Year's Day: January 1
    # Independence Day: July 4
    # Veterans Day: November 11
    # Christmas Day: December 25
    if month in [1, 7, 11, 12]:
        return 1
    
    # Thanksgiving: 4th Thursday of November
    # Labor Day: 1st Monday of September
    # Memorial Day: Last Monday of May
    # Martin Luther King Jr. Day: 3rd Monday of January
    # Presidents' Day: 3rd Monday of February
    # Columbus Day: 2nd Monday of October
    if month in [1, 2, 5, 9, 10, 11]:
        return 1
    
    return 0

def create_forecaster(reg, window_length=12, strategy='multioutput'):
    sklearn_pipeline = sklearn_make_pipeline(StandardScaler(), reg)
    m_reg = MultiOutputRegressor(sklearn_pipeline)
    return make_reduction(estimator=m_reg, window_length=window_length, strategy=strategy)


def train_model_mo(train, test, regressor, reg_name, win_in, win_out):
    X_train, y_train = train.iloc[:, :win_in], train.iloc[:, win_in:]
    X_test, y_test = test.iloc[:, :win_in], test.iloc[:, win_in:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(X_train)  
    y_train_s = scaler_y.fit_transform(y_train)  

    X_test_s = scaler_X.transform(X_test)
    y_test_s = scaler_y.transform(y_test)
    
    try:
        reg = regressor().fit(X_train_s, y_train_s) 
        print(f'Training {reg_name} ...')
    except:
        print(f'{reg_name} does not support multiple outputs')
        try:
            reg = MultiOutputRegressor(regressor()).fit(X_train_s, y_train_s)
            print(f'using sklearn MultipleOutput for {reg_name}')
        except Exception as e:
            print(f'Failed for {reg_name}: {e}')
        
    yhat = reg.predict(X_test_s)
    yhat = scaler_y.inverse_transform(yhat)
    actual = y_test.values

    # Compute evaluation metrics for multi-output, with multioutput='uniform_average'
    rmse_test = np.sqrt(mse(actual, yhat, multioutput='uniform_average'))
    mae_test = mae(actual, yhat, multioutput='uniform_average')
    mape_test = np.mean(np.abs((actual - yhat) / actual)) * 100 
    mase_test = mase(actual, yhat, y_train=y_train) 

    residuals = actual - yhat
    
    model_metadata = {
        'Model Name': reg_name, 
        'Model': reg, 
        'RMSE': rmse_test, 
        'MAPE': mape_test,
        'MASE': mase_test,
        'MAE': mae_test,
        'test': X_test,
        'yhat': pd.DataFrame(yhat, index=y_test.index, columns=y_test.columns), 
        'resid': pd.DataFrame(residuals, index=y_test.index, columns=y_test.columns),
        'actual': y_test
    }
    
    return model_metadata



from sklearn.multioutput import MultiOutputRegressor

def train_model_mo_sktime(train, test, reg, reg_name, strategy):
    
        
    fh = ForecastingHorizon(test.index, is_relative=False)
    sklearn_pipeline = sklearn_make_pipeline(StandardScaler(), reg)
    
    try:
        forecaster = make_reduction(estimator=sklearn_pipeline, 
                                window_length=10, 
                                strategy=strategy)
        forecaster.fit(train, fh=fh)
        print(f'training {reg_name} ...')
    except:
        try:
            mo_reg = MultiOutputRegressor(sklearn_pipeline)
            forecaster = make_reduction(estimator=mo_reg, 
                                window_length=10, 
                                strategy=strategy)
            forecaster.fit(train, fh=fh)
            print(f'using MultiOutputRegressor for {reg_name} ...')
        except Exception as e:
            print(f'Failed for {reg_name}: {e}')
        
    yhat = forecaster.predict(fh)
    actual = test.copy()
    
    rmse_test = np.sqrt(mse(actual, yhat))
    mae_test = mae(actual, yhat)
    mape_test = mape(actual, yhat)
    mase_test = mase(actual, yhat, y_train=(train))

    model_metadata = {
        'Model Name': reg_name, 
        'Model': forecaster, 
        'RMSE': rmse_test, 
        'MAPE': mape_test,
        'MASE': mase_test,
        'MAE': mae_test,
        'yhat': yhat, 
        'actual': actual}
    
    return model_metadata


def train_different_models_sktime(train, test, regressors, strategy):
    results = []
    for reg_name, regressor in regressors.items():
        reg = regressor()
        results.append(train_model_mo_sktime(train,
                                   test,
                                   reg,
                                   reg_name,
                                   strategy))
    return results


def train_model_sktime_cv(train, test, reg, reg_name, param_grid):

    # Full forecasting horizon for prediction
    fh = ForecastingHorizon(test.index, is_relative=False)
    
    # Multi-step horizon for CV
    fh_cv = np.arange(1, min(30, len(test) + 1))
    
    sklearn_pipeline = sklearn_make_pipeline(StandardScaler(), reg)

    # Define a 3-fold expanding window CV cross-validation strategy
    initial_window = max(50, len(train) // 3)
    remaining_length = len(train) - initial_window
    step_length = max(1, (remaining_length - len(test)) // 2)  
    
    cv = ExpandingWindowSplitter(initial_window=initial_window, 
                                 step_length=step_length, 
                                 fh=fh_cv)
    
    try:
        # Wrap the regressor in a MultiOutputRegressor
        m_reg = MultiOutputRegressor(sklearn_pipeline)
        forecaster = make_reduction(estimator=m_reg, 
                                window_length=10, # 10 lags, adjustable
                                strategy="multioutput")

        # Perform grid search with cross-validation
        gscv = ForecastingGridSearchCV(
                        forecaster=forecaster,
                        param_grid=param_grid,
                        cv=cv,
                        verbose=1)

        gscv.fit(train, fh=fh)
        print(f'Grid Search CV using MultiOutputRegressor for {reg_name} ...')
    except Exception as e:
        print(f'Failed for {reg_name}: {e}')
        return
    
    # Generate predictions and evaluate
    yhat = gscv.predict(fh)
    actual = test.copy()
    
    # Evaluate model performance using various metrics     
    rmse_test = np.sqrt(mse(actual, yhat))
    mae_test = mae(actual, yhat)
    mape_test = mape(actual, yhat)
    mase_test = mase(actual, yhat, y_train=(train))

    model_metadata = {
        'Model Name': reg_name, 
        'Model': gscv, 
        'RMSE': rmse_test, 
        'MAPE': mape_test,
        'MASE': mase_test,
        'MAE': mae_test,
        'yhat': yhat, 
        'actual': actual}
    
    return model_metadata

def train_different_models_sktime_cv(train, test, regressors):
    results = []
    for reg_name, regressor in regressors.items():
        reg = regressor[0]()
        param_grid = regressor[1]
        results.append(train_model_sktime_cv(train,
                                   test,
                                   reg,
                                   reg_name, 
                                   param_grid))
    return results