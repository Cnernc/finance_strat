import numpy as np
import pandas as pd

##### Stock Data Management Functions

# This section contains functions for managing stock market data. 
# These functions allow for downloading, processing, and handling 
# stock price data from Yahoo Finance. Functions include capabilities 
# for downloading data directly from Yahoo Finance, saving it locally 
# for efficient access, and preparing the data for further analysis. 
# This ensures a streamlined process for handling stock data, 
# from acquisition to preprocessing.

import yfinance as yf
import os

def download_data_from_yf(tickers, start_date, end_date, price_types=['Open']) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance for specified tickers and date range.
    
    Parameters:
    - tickers (list): List of stock tickers.
    - start_date (str): Start date for data download.
    - end_date (str): End date for data download.
    - price_types (list): Types of prices to download (e.g., 'Open', 'Close'). Defaults to ['Open'].

    Returns:
    - pd.DataFrame: DataFrame with stock prices.
    """
    
    # Download stock data for each ticker within the specified date range
    data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

    # Initialize a DataFrame to store price data
    prices_df = pd.DataFrame()

    # Populate the DataFrame with selected price types for each ticker
    for ticker in tickers:
        for price_type in price_types:
            prices_df[f'{ticker}'] = data[ticker][price_type]

    # Fill missing values with the last available value
    prices_df = prices_df.resample('1B').ffill().fillna(0)

    return prices_df

def get_or_download_data(tickers, start_date, end_date, price_types=['Open'], directory='stock_data') -> pd.DataFrame:
    """
    Get or download stock data for specified tickers and date range. 
    Downloads data from Yahoo Finance if not already saved locally, else reads from saved files.
    A separate file is created for each ticker.
    Data is combined and any missing values are forward-filled at the end.

    Parameters:
    - tickers (list): List of stock tickers.
    - start_date (str): Start date for data download.
    - end_date (str): End date for data download.
    - price_types (list): Types of prices to download (e.g., 'Open', 'Close'). Defaults to ['Open'].
    - directory (str): Directory to save/read the ticker files. Defaults to 'stock_data'.

    Returns:
    - pd.DataFrame: DataFrame with stock prices for all tickers.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    prices_df_all = pd.DataFrame()

    for ticker in tickers:
        file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
        file_path = os.path.join(directory, file_name)

        if os.path.exists(file_path):
            price_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            # Download data for the ticker and save to file
            data = yf.download(ticker, start=start_date, end=end_date)
            price_df = pd.DataFrame({f'{ticker} {price_type}': data[price_type] for price_type in price_types})
            price_df.to_csv(file_path)

        # Combine data without filling missing values yet
        prices_df_all = pd.concat([prices_df_all, price_df], axis=1)

    # Forward-fill missing values at the end
    prices_df_all = prices_df_all.resample('1B').ffill().fillna(0)

    return prices_df_all

##########    ##########    ##########    ##########    ##########
##### Visualization Functions

# This section includes functions designed for visualizing financial data. 
# Utilizing libraries like Matplotlib and Seaborn, these functions provide 
# tools to plot and interpret stock market data in a clear and informative way. 
# From basic line plots to more complex visualizations, these functions 
# help in making data-driven insights more accessible and understandable.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_df(df: pd.DataFrame, title: str = '', unit: str = '', scaled: bool = False) -> None:
    """
    Plot the given DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - title (str): Title for the plot.
    - unit (str): Unit of measurement for the y-axis.
    - scaled (bool): If True, scale data before plotting.
    """
    local_df = df.copy()
        
    plt.figure(figsize=(10, 5))
    plt.grid() 
    
    # Scale data if required
    if scaled: 
        local_df = (local_df - local_df.mean()) / local_df.std() 
        plt.title(title + ' (scaled)')
        unit = '(std)'
    else: 
        plt.title(title)

    plt.xlabel('Date')
    plt.ylabel(f'{title} {unit}')

    # Plot each column in the DataFrame
    for col in local_df.columns: 
        sns.lineplot(x=local_df.index, y=local_df[col], label=col)
    
    plt.legend()
    plt.show()

def plot_ds_hue(ds: pd.Series, ds_hue: pd.Series, title: str = ''):
    """
    Plot a Series with points colored based on another Series.
    
    Parameters:
    - ds (pd.Series): Series to plot.
    - ds_hue (pd.Series): Series used to color the points.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('USD (scaled on: 25kUSD / std risk)')

    # Warn if the Series is constant
    if ds.std() == 0: 
        print(f'Warning: ds, {ds} is constant.')
        return
    elif ds_hue.std() == 0: 
        print(f'Warning: ds_hue, {ds_hue} is constant.')
        return
    
    # Scale the Series
    ds_scaled = 25 * 1000 * ds / ds.std()

    # Plot daily and cumulative lines
    sns.lineplot(x=ds_scaled.index, y=ds_scaled, color="lightblue", linewidth=0.7, label='Daily')
    sns.lineplot(x=ds_scaled.index, y=ds_scaled.cumsum(), color="grey", linewidth=0.7, label='Cumulative')

    # Scale the hue data
    window_size = 100 
    ds_hue_scaled = (ds_hue - ds_hue.rolling(window=window_size).mean()) / (ds_hue.rolling(window=window_size).std() + 1e-8)
    ds_hue_scaled = ds_hue_scaled.abs()
    ds_hue_scaled.fillna(0, inplace=True)

    # Cap extreme values and plot as scatter
    ds_hue_filtered = np.where(ds_hue_scaled.abs() > 2, np.sign(ds_hue_scaled) * 2, ds_hue_scaled)
    sns.scatterplot(x=ds_scaled.index, y=ds_scaled.cumsum(), 
                    hue=ds_hue_filtered, size=np.abs(ds_hue_filtered), 
                    palette="coolwarm", hue_order=[0, 1, 2], label='Signal')
    plt.legend()
    plt.show()

##########    ##########    ##########    ##########    ##########
##### Advanced Financial Analysis Functions

# This section comprises functions for advanced financial analysis. 
# These functions are essential for computing key financial indicators like Z-Scores, 
# determining trading positions based on market signals, adjusting positions for volatility, 
# and calculating profit and loss. They are crucial for quantitative trading strategies, 
# enabling a detailed and nuanced understanding of market dynamics and portfolio performance.

def moving_Zscore(df: pd.DataFrame, window_size_zs: int = 30) -> pd.DataFrame:
    """
    Calculate the Z-Score for each column in a DataFrame over a rolling window.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to calculate Z-Scores for.
    - window_size_zs (int): Size of the rolling window for Z-Score calculation.

    Returns:
    - pd.DataFrame: DataFrame of Z-Scores.
    """
    # Return original DataFrame or DataFrame of 1s based on the window size
    if window_size_zs == 0: 
        return df
    if window_size_zs == np.inf: 
        return df.map(lambda x: 1)

    # Calculate Z-Score using rolling mean and standard deviation
    m_zscore = (df - df.rolling(window=window_size_zs).mean()) / df.rolling(window=window_size_zs).std()
    
    return m_zscore

def compute_position(signals_df: pd.DataFrame, prices_df: pd.DataFrame, window_size: int = 50, threshold: int = 0) -> pd.DataFrame:
    """
    Compute trading positions based on signals and stock prices.

    Parameters:
    - signals (pd.DataFrame): DataFrame containing trading signals.
    - prices (pd.DataFrame): DataFrame containing stock prices.
    - window_size (int): Size of the rolling window for signal processing.
    - threshold (int): Threshold for signal strength to take a position.

    Returns:
    - pd.DataFrame: DataFrame containing trading positions.
    """
    # Initialize a DataFrame to store positions
    positions_df = pd.DataFrame(index=signals_df.index)
    
    # Calculate rolling mean and standard deviation for signals
    rolling_mean = signals_df.rolling(window=window_size).mean()
    rolling_std = signals_df.rolling(window=window_size).std()
    
    # Compute Z-scores for signals and set positions where signals exceed the threshold
    zscore_signal = (signals_df - rolling_mean) / rolling_std
    zscore_signal.fillna(0, inplace=True)
    positions_df = prices_df * signals_df.where(zscore_signal.abs() > threshold, 0)

    return positions_df

def adjust_position_volatility_targetting(positions_unadjusted: pd.DataFrame, window_size_vt: int = 0) -> pd.DataFrame:
    """
    Adjust trading positions for volatility targeting.

    Parameters:
    - positions_unadjusted (pd.DataFrame): DataFrame containing original trading positions.
    - window_size_vt (int): Rolling window size for volatility calculation.

    Returns:
    - pd.DataFrame: DataFrame containing volatility-adjusted trading positions.
    """
    # Return original positions if no volatility targeting is specified
    if window_size_vt in (0, np.inf):
        return positions_unadjusted

    # Initialize a DataFrame for adjusted positions
    positions_adjusted = pd.DataFrame(index=positions_unadjusted.index)

    # Calculate rolling volatility and adjust positions accordingly
    volatility_df = positions_unadjusted.rolling(window=window_size_vt).std()
    volatility_df.replace(0, 1, inplace=True)  # Avoid division by zero
    for col in positions_unadjusted.columns:
        positions_adjusted[col] = positions_unadjusted[col] / volatility_df[col]

    return positions_adjusted

def compute_PNL(positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Profit and Loss (PNL) from trading positions and price changes.

    Parameters:
    - positions (pd.DataFrame): DataFrame containing trading positions.
    - prices (pd.DataFrame): DataFrame containing stock prices.

    Returns:
    - pd.DataFrame: DataFrame containing daily and cumulative PNL.
    """

    # Initialize a DataFrame for PNL
    pnl_df = pd.DataFrame(index=prices_df.index)
    
    # Calculate daily PNL based on position changes and price movements
    pnl_df["DAILY"] = sum([positions_df[ticker].shift(1) * prices_df[ticker].pct_change() for ticker in prices_df.columns])
    pnl_df["CUMULATIVE"] = pnl_df["DAILY"].cumsum()

    return pnl_df

##########    ##########    ##########    ##########    ##########
##### Performance Evaluation Functions

# This section includes functions specifically designed for evaluating the performance of trading strategies. 
# These functions calculate key metrics such as the Sharpe Ratio, turnover, and basis points (bips), 
# essential for assessing the effectiveness and efficiency of trading activities. 
# They provide a comprehensive view of the strategy's risk-adjusted returns, 
# enabling a deeper analysis of investment decisions and market behavior.

def compute_metrics(pnl_df: pd.DataFrame, positions_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute various performance metrics for the trading strategy.

    Parameters:
    - pnl (pd.DataFrame): DataFrame containing daily PNL data.
    - positions (pd.DataFrame): DataFrame containing trading positions.

    Returns:
    - dict[str, float]: Dictionary containing key performance metrics.
    """
    # Calculate daily pnl and absolute positions
    pnl_daily = pnl_df['DAILY']
    positions_abs_sum = positions_df.abs().sum(axis=1)

    # Compute Sharpe Ratio and other metrics
    sharpe_ratio = np.sqrt(252) * pnl_daily.mean() / pnl_daily.std()
    turnover = 100 * positions_abs_sum.diff().abs().mean() / positions_abs_sum.abs().mean()
    bips = 100 * 100 * pnl_daily.mean() / positions_abs_sum.diff().abs().mean()

    # Compute effective metrics considering only non-zero positions
    positions_data_eff = positions_abs_sum.where(positions_df.abs().sum(axis=1) != 0)
    pnl_daily_eff = pnl_daily.where(positions_df.abs().sum(axis=1) != 0)
    eff_sharpe_ratio = np.sqrt(252) * pnl_daily_eff.mean() / pnl_daily_eff.std()
    eff_bips = 100 * 100 * pnl_daily_eff.mean() / positions_abs_sum.diff().abs().mean()
    eff_turnover = 100 * positions_data_eff.diff().abs().mean() / positions_data_eff.abs().mean()

    # Return metrics as a dictionary
    return {
        'sharpe_ratio': round(sharpe_ratio, 2), 
        'turnover': round(turnover, 1), 
        'bips': round(bips, 1), 
        'effective_sharpe_ratio': round(eff_sharpe_ratio, 2), 
        'eff_bips': round(eff_bips, 2), 
        'eff_turnover': round(eff_turnover, 2)
    }
