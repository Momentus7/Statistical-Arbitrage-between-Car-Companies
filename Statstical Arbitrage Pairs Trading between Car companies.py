#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(0)
time = np.arange(0, 100, 1)
price = np.random.normal(0, 1, 100).cumsum()
price[price < 0] = 0
plt.figure(figsize=(10, 6))
plt.plot(time, price, label='Stock Price')
point1 = random.randint(0, len(price) - 2)
possible_points = [i for i in range(point1 + 1, len(price)) if price[i] < price[point1]]
if not possible_points:
    point2 = point1 + 1
else:
    point2 = random.choice(possible_points)

plt.scatter([time[point1], time[point2]], [price[point1], price[point2]], color='green')
plt.text(time[point1], price[point1], f'({price[point1]:.2f})', color='green')
plt.text(time[point2], price[point2], f'({price[point2]:.2f})', color='green')

plt.title('Profit on a Short Position')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(0)
time = np.arange(0, 100, 1)
price = np.random.normal(0, 1, 100).cumsum()
price[price < 0] = 0
plt.figure(figsize=(10, 6))
plt.plot(time, price, label='Stock Price')
point1 = random.randint(0, len(price) - 2)
possible_points = [i for i in range(point1 + 1, len(price)) if price[i] > price[point1]]
if not possible_points:
    point2 = point1 + 1
else:
    point2 = random.choice(possible_points)

plt.scatter([time[point1], time[point2]], [price[point1], price[point2]], color='green')
plt.text(time[point1], price[point1], f'({price[point1]:.2f})', color='green')
plt.text(time[point2], price[point2], f'({price[point2]:.2f})', color='green')

plt.title('Loss on a Short Position')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(0)
time = np.arange(0, 100, 1)
price = np.random.normal(0, 1, 100).cumsum()
price[price < 0] = 0
plt.figure(figsize=(10, 6))
plt.plot(time, price, label='Stock Price')
point1 = random.randint(0, len(price) - 2)
possible_points = [i for i in range(point1 + 1, len(price)) if price[i] > price[point1]]
if not possible_points:
    point2 = point1 + 1
else:
    point2 = random.choice(possible_points)

plt.scatter([time[point1], time[point2]], [price[point1], price[point2]], color='green')
plt.text(time[point1], price[point1], f'({price[point1]:.2f})', color='green')
plt.text(time[point2], price[point2], f'({price[point2]:.2f})', color='green')

plt.title('Profit on a Long Position')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(0)
time = np.arange(0, 100, 1)
price = np.random.normal(0, 1, 100).cumsum()
price[price < 0] = 0
plt.figure(figsize=(10, 6))
plt.plot(time, price, label='Stock Price')
point1 = random.randint(0, len(price) - 2)
possible_points = [i for i in range(point1 + 1, len(price)) if price[i] > price[point1]]
if not possible_points:
    point2 = point1 + 1
else:
    point2 = random.choice(possible_points)

plt.scatter([time[point1], time[point2]], [price[point1], price[point2]], color='green')
plt.text(time[point1], price[point1], f'({price[point1]:.2f})', color='green')
plt.text(time[point2], price[point2], f'({price[point2]:.2f})', color='green')

plt.title('Loss on a Long Position')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# #A Pairwise Engle-Granger test and an Augmented Dickey-Fuller (ADF) test.

# In[10]:


import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) 


stocks = ['TM', 'GM', 'F', 'HMC', 'NSANY', 'HYMTF']

stock_data = {stock: yf.download(stock, period='10y')['Close'] for stock in stocks}


p_values = pd.DataFrame(index=stocks, columns=stocks)


for pair in itertools.combinations(stocks, 2):
    
    Y = stock_data[pair[0]].dropna()
    X = stock_data[pair[1]][Y.index]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    residuals = model.resid

    
    adf_result = adfuller(residuals)
    p_value = adf_result[1]

    
    p_values.loc[pair[0], pair[1]] = p_value
    p_values.loc[pair[1], pair[0]] = p_value


def highlight_significant(val):
    color = 'green' if val < 0.05 else 'red'
    return 'color: %s' % color
styled_p_values = p_values.style.applymap(highlight_significant)
styled_p_values


# In[11]:


hmc_data = yf.download('HMC', period='10y')['Close']
hymtf_data = yf.download('HYMTF', period='10y')['Close']


spread = hymtf_data - hmc_data


adf_result = adfuller(spread.dropna())


print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
for key, value in adf_result[4].items():
    print(f'Critical Value ({key}): {value}')


# In[12]:


import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.plot(hmc_data, label='HMC')
plt.plot(hymtf_data, label='HYMTF')
plt.title('HMC vs HYMTF (10 Years)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))
plt.plot(spread, label='Spread')
plt.title('Spread between HMC & HYMTF (10 Years)')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


from scipy.stats import spearmanr


hmc_returns = hmc_data.pct_change().dropna()
hymtf_returns = hymtf_data.pct_change().dropna()


spearman_corr, p_value = spearmanr(hmc_returns, hymtf_returns)


print(f"Spearman Rank Correlation Coefficient: {spearman_corr:.3f}")
print(f"P-value: {p_value:.3f}")


# In[15]:


# Backtesting


# In[16]:


import yfinance as yf
import pandas as pd


hmc_data = yf.download('HMC', period='10y')['Close']
hymtf_data = yf.download('HYMTF', period='10y')['Close']


spread = hymtf_data - hmc_data


rolling_window = 8     
rolling_mean = (spread.rolling(window=rolling_window).mean()).dropna()
rolling_std = (spread.rolling(window=rolling_window).std()).dropna()
upper_band = rolling_mean + (rolling_std * 0.8)
lower_band = rolling_mean - (rolling_std * 0.8)


# In[17]:


money_per_trade = 50000


hmc_share_quant = money_per_trade/hmc_data
hymtf_share_quant = money_per_trade/hymtf_data

daily_returns = []
trade_log = []
position_open = False
position_type = None

for date, current_spread in spread.items():
    if date not in hmc_data.index or date not in upper_band.index:
        continue
    hmc_price = hmc_data.get(date, None)
    hymtf_price = hymtf_data.get(date, None)
    lower_band_value = lower_band[date]
    upper_band_value = upper_band[date]
    mean_value = rolling_mean[date]
    hmc_shares = hmc_share_quant.get(date, None)
    hymtf_shares = hymtf_share_quant.get(date, None)
    if current_spread < lower_band_value and not position_open:
        # Open a pairs trade position (short HMC, long HYMTF)
        trade_log.append({
            'Date': date,
            'Action': 'Open Position',
            'Type': 'lower',
            'HMC_Shares': hmc_shares,
            'HYMTF_Shares': hymtf_shares,
            'HMC_Price': hmc_price,
            'HYMTF_Price': hymtf_price
        })
        position_open = True
        position_type = 'lower'
        
    elif current_spread > upper_band_value and not position_open:
        
        trade_log.append({
            'Date': date,
            'Action': 'Open Position',
            'Type': 'upper',
            'HMC_Shares': hmc_shares,
            'HYMTF_Shares': hymtf_shares,
            'HMC_Price': hmc_price,
            'HYMTF_Price': hymtf_price
        })
        position_open = True
        position_type = 'upper'
    
    elif current_spread > mean_value and position_open and position_type == 'lower':
        
        trade_log.append({
            'Date': date,
            'Action': 'Close Position',
            'Type': 'lower',
            'HMC_Shares': hmc_shares,
            'HYMTF_Shares': hymtf_shares,
            'HMC_Price': hmc_price,
            'HYMTF_Price': hymtf_price
        })
        position_open = False
        
    elif current_spread < mean_value and position_open and position_type == 'upper':
        
        trade_log.append({
            'Date': date,
            'Action': 'Close Position',
            'Type': 'upper',
            'HMC_Shares': hmc_shares,
            'HYMTF_Shares': hymtf_shares,
            'HMC_Price': hmc_price,
            'HYMTF_Price': hymtf_price
        })
        position_open = False


trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv('tradelog.csv')


# In[18]:


total_return = 0
open_position = None
close_position_dates = []
close_position_returns = []
for index, trade in trade_log_df.iterrows():
    if trade['Action'] == 'Open Position':
        open_position = trade
    elif trade['Action'] == 'Close Position' and open_position is not None:
        close_position = trade
        close_position_dates.append(trade['Date'])
        if open_position['Type'] == 'lower':
            hmc_profit_loss = (open_position['HMC_Price'] - trade['HMC_Price']) * (open_position['HMC_Shares'])
            hymtf_profit_loss = (trade['HYMTF_Price'] - open_position['HYMTF_Price']) * (open_position['HYMTF_Shares'])
        elif open_position['Type'] == 'upper':
            hmc_profit_loss = (trade['HMC_Price'] - open_position['HMC_Price']) * (open_position['HMC_Shares'])
            hymtf_profit_loss = (open_position['HYMTF_Price'] - trade['HYMTF_Price']) * (open_position['HYMTF_Shares'])

        total_profit_loss = hmc_profit_loss + hymtf_profit_loss
        total_return += total_profit_loss
        close_position_returns.append(total_profit_loss)

profit_loss_df = pd.DataFrame(close_position_returns, index=close_position_dates, columns=['Profit/Loss'])
profit_loss_df.to_csv('returns.csv')


# In[19]:


import matplotlib.pyplot as plt


profit_loss_df['Cumulative PnL'] = profit_loss_df['Profit/Loss'].cumsum()


plt.figure(figsize=(10, 6))
plt.plot(profit_loss_df.index, profit_loss_df['Cumulative PnL'], marker='o', linestyle='-')
plt.title('Realized Cumulative Profit/Loss Chart')
plt.xlabel('Close Position Dates')
plt.ylabel('Cumulative Profit/Loss')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


capital = 100000  
profit_loss_df['Returns'] = profit_loss_df['Profit/Loss'] / capital


profit_loss_df['Year'] = profit_loss_df.index.year


trades_per_year = profit_loss_df.groupby('Year').size()
average_trades_per_year = trades_per_year.mean()


annual_returns = profit_loss_df.groupby('Year')['Returns'].sum()
annual_std_dev = profit_loss_df.groupby('Year')['Returns'].std() * np.sqrt(average_trades_per_year)
annual_risk_free_rate = 0.01


annual_sharpe_ratio = (annual_returns - annual_risk_free_rate) / annual_std_dev


average_sharpe_ratio = annual_sharpe_ratio.mean()


print("Annual Sharpe Ratio:\n", annual_sharpe_ratio)
print("\nNumber of Trades per Year:\n", trades_per_year)
print("\nAverage Sharpe Ratio:", average_sharpe_ratio)

plt.figure(figsize=(12, 6))
plt.plot(annual_sharpe_ratio, label='Annual Sharpe Ratio', marker='o')
plt.axhline(y=average_sharpe_ratio, color='r', linestyle='--', label='Average Sharpe Ratio')
plt.title('Annual Sharpe Ratio with Average')
plt.xlabel('Year')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show()

