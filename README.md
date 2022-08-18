# web-scraping-machine-learning
Web Scraping for Machine Learning


```bash
$ python3 -m pip install requests_html beautifulsoup4
```

```bash
$ python3 -m pip install pandas numpy matplotlib seaborn tensorflow sklearn
```

```python
from requests_html import HTMLSession
import pandas as pd
```

```python
url = 'http://your-target-url'
session = HTMLSession()
r = session.get(url)
```

```python
rows = r.html.xpath('//table/tbody/tr')
symbol = 'AAPL'
data = []
for row in rows:
    if len(row.xpath('.//td')) < 7:
        continue
    data.append({
        'Symbol':symbol,
        'Date':row.xpath('.//td[1]/span/text()')[0],
        'Open':row.xpath('.//td[2]/span/text()')[0],
        'High':row.xpath('.//td[3]/span/text()')[0],
        'Low':row.xpath('.//td[4]/span/text()')[0],
        'Close':row.xpath('.//td[5]/span/text()')[0],
        'Adj Close':row.xpath('.//td[6]/span/text()')[0],
        'Volume':row.xpath('.//td[7]/span/text()')[0]
    }) 
```

```python
df['Date'] = pd.to_datetime(df['Date'])
```

```python
str_cols = ['High', 'Low', 'Close', 'Adj Close', 'Volume']
df[str_cols]=df[str_cols].replace(',', '', regex=True).astype(float)
```

```python
df.dropna(inplace=True)
```

```python
df = df.set_index('Date')
df.head()
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.style.use("ggplot")
```

```python
plt.figure(figsize=(15, 6))
df['Adj Close'].plot()
plt.ylabel('Adj Close')
plt.xlabel(None)
plt.title('Closing Price of AAPL')
```

```python
features = ['Open', 'High', 'Low', 'Volume']
y = df.filter(['Adj Close'])
```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
```

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=10) 
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

```python
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
```

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
```

```python
model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

```python
model.fit(X_train, y_train, epochs=100, batch_size=8)
```

```python
y_pred= model.predict(X_test)
```

```python
plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Actual Value')
plt.plot(y_pred, label='Predicted Value')
plt.ylabel('Adjusted Close (Scaled)')
plt.xlabel('Time Scale')
plt.legend()
```

