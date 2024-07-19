import pandas as pd
import numpy as np

from drawdowns import rolling_max_dd


required_columns = ['Date', 'Symbol', 'Price', 'Size', 'Side']
rolling_window = int(365.25/12)  # days per month
annual_trading_days = 252  # days


def get_ticker_price(number, seed=None) -> float:
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(100, 500, number)


def check_all_cols(df: pd.DataFrame):
    # Check if all required columns are in the DataFrame
    if all(column in df.columns for column in required_columns):
        print("All required columns are present.")
    else:
        print("Some required columns are missing.")


def handle_missing_values(new_df: pd.DataFrame) -> pd.DataFrame:
    # Handling missing values
    # Dropping rows where 'Date', 'Symbol', 'Side' or "Price" is missing
    new_df.dropna(subset=['Date', 'Symbol', 'Side', 'Price'], inplace=True)

    # Filling 'Price' with rand and 'Size' with 1
    new_df['Size'].fillna(1, inplace=True)

    return new_df


def long_or_short(trades_data):
    labeled_trades = pd.DataFrame()

    # Loop through each symbol in the DataFrame
    for symbol, group in trades_data.groupby('Symbol'):
        group = group.reset_index(drop=True)
        group['Type'] = 'long'  # default trade label to long

        group['Closing Value'] = group['Closing Price'] * group['Size'].cumsum()

        # Apply the short selling logic
        for i in range(len(group) - 1):
            if 'sell' in group.iloc[i]['Side']:
                sell_price = group.iloc[i]['Price']
                for j in range(i + 1, len(group)):
                    if 'buy' in group.iloc[j]['Side'] and group.iloc[j]['Price'] < sell_price:
                        group.loc[[i, j], 'Type'] = 'short'
                    else:
                        break

        # Append the processed group back into the labeled_trades DataFrame
        labeled_trades = pd.concat([labeled_trades, group])

    labeled_trades = labeled_trades.sort_values('Date').reset_index(drop=True)

    return labeled_trades


def equity_curve(labeled_trades):
    labeled_trades['Date'] = pd.to_datetime(labeled_trades['Date'])

    # Find all unique symbols
    symbols = labeled_trades['Symbol'].unique()

    # Initialize a dictionary to hold the reindexed DataFrames
    reindexed_data = {}

    # Find the union of all trading dates across all securities
    all_dates = pd.date_range(start=labeled_trades['Date'].min(), end=labeled_trades['Date'].max())

    # Process each symbol
    for symbol in symbols:
        # Isolate the security data
        security_data = labeled_trades[labeled_trades['Symbol'] == symbol].copy().reset_index(drop=True)
        security_data.set_index('Date', inplace=True)

        # Reindex and interpolate within their trading ranges. I could also use the random number generator.
        # In real life, pull the closing value from history/market data
        interpolated = security_data[['Closing Value']].reindex(all_dates).interpolate(method='time')
        interpolated = interpolated.loc[security_data.index.min():security_data.index.max()]

        # Store the reindexed DataFrame
        reindexed_data[symbol] = interpolated

    # Combine the DataFrames
    combined = pd.concat(reindexed_data, axis=1)
    combined.columns = combined.columns.droplevel(1)  # Clean up the column headers

    # Sum up the closing values across securities for each date
    combined['Total value'] = combined.sum(axis=1)

    # Reset the index to make 'Date' a column again, if you want to manipulate or display it
    combined.reset_index(inplace=True)
    combined.rename(columns={'index': 'Date'}, inplace=True)
    return combined


def evaluate_cagr(trade_df, equity_df):
    start_date = trade_df['Date'].min()
    end_date = trade_df['Date'].max()
    years = (end_date - start_date).days / 365.25  # to account for leap years

    total_investment = trade_df['Cost'].sum()
    final_value = equity_df[equity_df['Date'] == end_date]['Total value'].reset_index(drop=True)
    cagr = ((final_value.values[0] / total_investment) ** (1 / years) - 1) * 100

    return round(cagr, 2)


def get_metric_long_short(df, target_col, sum_or_mean='sum'):
    long_trades = df[df['Type'] == 'long']
    short_trades = df[df['Type'] == 'short']

    if sum_or_mean == 'sum':
        metric_total = df[target_col].sum()
        metric_long = long_trades[target_col].sum()
        metric_short = short_trades[target_col].sum()
    else:
        metric_total = df[target_col].mean()
        metric_long = long_trades[target_col].mean()
        metric_short = short_trades[target_col].mean()

    return metric_total, metric_long, metric_short


def evaluate_roi(merged_df):
    roi = merged_df[['Type', 'Cost', 'Total value']].copy()

    roi['rolling cost'] = roi['Cost'].cumsum()
    roi['ROI'] = (roi['Total value'] - roi['rolling cost']) / roi['rolling cost'] * 100
    roi = roi[roi['ROI'] != roi['ROI'].min()]

    roi_long = roi[roi['Type'] == 'long']
    roi_short = roi[roi['Type'] == 'short']

    return roi, roi_long, roi_short


def rolling_sharpe(equity, window_size):
    risk_free_rate = 0.03  # Assume a risk-free rate of 3%

    rolling = pd.DataFrame()
    rolling['Date'] = equity['Date'].copy()
    rolling['Rolling sharpe'] = equity['Total value'].pct_change(fill_method=None).copy()
    rolling = rolling.dropna().reset_index(drop=True)

    mean = rolling['Rolling sharpe'].rolling(window=window_size).mean()
    std = rolling['Rolling sharpe'].rolling(window=window_size).std()
    rolling['Rolling sharpe'] = (mean - risk_free_rate) / std * np.sqrt(annual_trading_days)
    rolling = rolling.dropna().reset_index(drop=True)
    return rolling


def annual_vol_return(equity):
    daily_returns = equity.drop(columns=['Date']).pct_change(fill_method=None)

    returns = daily_returns.mean() * annual_trading_days
    volatility = daily_returns.std() * annual_trading_days**0.5
    return returns, volatility


def find_correlation(equities):
    daily_returns = equities.drop(columns=['Date', 'Total value']).pct_change(fill_method=None)
    daily_returns.dropna(inplace=True)

    # Calculate the correlation matrix from the daily returns
    correlation_matrix = daily_returns.corr()

    # Count only the upper triangle excluding the diagonal
    upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
    positive_correlations = np.sum(correlation_matrix.values[upper_triangle_indices] > 0)
    negative_correlations = np.sum(correlation_matrix.values[upper_triangle_indices] < 0)
    pos_neg_corr_ratio = positive_correlations / negative_correlations

    string = "Portfolio is diverse.\n" if pos_neg_corr_ratio < 1 else "Portfolio is not diverse.\n"

    return correlation_matrix, round(pos_neg_corr_ratio, 2), string


def calculate_trade_performance(trades_df: pd.DataFrame) -> dict:
    metrics = {}
    if trades_df.empty:
        print(f"Input dataframe is empty. Try again with non-empty dataframe.")
        return metrics  # Return empty dictionary for empty DataFrame
    else:
        new_df = trades_df.copy()

    check_all_cols(new_df)
    new_df = handle_missing_values(new_df)

    # Sort rows by date to establish chronological order
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df.sort_values('Date').reset_index(drop=True)

    # assign 1 to buy side and -1 to sell side
    new_df['Size'] = new_df['Size'] * np.where(new_df['Side'] == 'buy', 1, -1)

    # cost involved in transaction
    new_df['Cost'] = new_df['Price'] * new_df['Size']

    # Calculate the current market value
    new_df['Closing Price'] = get_ticker_price(len(new_df), seed=50)

    new_df = long_or_short(new_df)

    # equity curve
    equity_df = equity_curve(new_df)
    metrics['Equity curves'] = equity_df

    # CAGR
    metrics['CAGR'] = evaluate_cagr(new_df, equity_df)

    # rolling maximum drawdown
    metrics['RMDD'] = rolling_max_dd(equity_df['Total value'], window_size=rolling_window)/1e6 # normalized per M

    # First, merge the trade execution data with market values on dates and symbols
    merged_df = pd.merge(new_df, equity_df, how='left', on='Date')
    merged_df.set_index('Date', inplace=True)

    # Calculate net profit/loss for each trade
    # Use closing price from market value data for calculations
    merged_df['Net P/L'] = (merged_df['Closing Price'] - merged_df['Price']) * merged_df['Size']

    # Calculate total net profit/loss
    net_pnl, net_pnl_long, net_pnl_short = get_metric_long_short(merged_df, 'Net P/L')
    metrics['Net P/L in $'] = {'all': net_pnl, 'long': net_pnl_long, 'short': net_pnl_short}

    # # profit factor
    gross_profit_df = merged_df[merged_df['Net P/L'] > 0]
    gross_loss_df = merged_df[merged_df['Net P/L'] < 0]
    gp, gp_long, gp_short = get_metric_long_short(gross_profit_df, 'Net P/L')
    gl, gl_long, gl_short = get_metric_long_short(gross_loss_df, 'Net P/L')
    pf, pf_long, pf_short = gp / abs(gl), gp_long / abs(gl_long), gp_short / abs(gl_short)
    metrics['Profit factor'] = {'all': pf, 'long': pf_long, 'short': pf_short}

    # # win-to-loss ratio
    avg_win, avg_win_long, avg_win_short = get_metric_long_short(gross_profit_df, 'Net P/L', sum_or_mean='mean')
    avg_loss, avg_loss_long, avg_loss_short = get_metric_long_short(gross_loss_df, 'Net P/L', sum_or_mean='mean')
    w2l, w2l_long, w2l_short = avg_win / abs(avg_loss), avg_win_long / abs(avg_loss_long), avg_win_short / abs(
        avg_loss_short)
    metrics['Win-to-loss ratio'] = {'all': w2l, 'long': w2l_long, 'short': w2l_short}

    # # ROI
    ROI, roi_long, roi_short = evaluate_roi(merged_df)
    w2l_ROI_short = (roi_short['ROI'] > 0).mean() / (roi_short['ROI'] < 0).mean()
    w2l_ROI_long = (roi_long['ROI'] > 0).mean() / (roi_long['ROI'] < 0).mean()
    window = 3  # at every 3rd data point
    roi_short = roi_short['ROI'].rolling(window=window).mean()
    roi_long = roi_long['ROI'].rolling(window=window).mean()
    metrics['ROI'] = {'all': ROI, 'long': roi_long, 'short': roi_short,
                      'w2l_ROI_short': w2l_ROI_short, 'w2l_ROI_long': w2l_ROI_long}

    # Sharpe Ratio
    metrics['Rolling Sharpe ratio'] = rolling_sharpe(equity_df, 126)

    # returns & vols
    annualized_returns, annualized_volatility = annual_vol_return(equity_df)
    metrics['Annualized volatility'] = annualized_volatility
    metrics['Annualized returns'] = annualized_returns

    # correlation
    correlation_matrix, pos_neg_corr_ratio, string = find_correlation(equity_df)
    metrics['Correlation'] = {'matrix': correlation_matrix, 'verdict': string,
                              'pos_neg_corr_ratio': pos_neg_corr_ratio}

    return metrics
