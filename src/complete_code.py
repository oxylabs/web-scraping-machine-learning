from requests_html import HTMLSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense

url = 'https://finance.yahoo.com/quote/AAPL/history?p=AAPL&guccounter=1&period1=1556113078&period2=1713965616'
session = HTMLSession()
r = session.get(url)

rows = r.html.xpath('//table/tbody/tr')
symbol = 'AAPL'
data = []
for row in rows:
    if len(row.xpath('.//td')) < 7:
        continue
    data.append({
        'Symbol':symbol,
        'Date':row.xpath('.//td[1]/text()')[0],
        'Open':row.xpath('.//td[2]/text()')[0],
        'High':row.xpath('.//td[3]/text()')[0],
        'Low':row.xpath('.//td[4]/text()')[0],
        'Close':row.xpath('.//td[5]/text()')[0],
        'Adj Close':row.xpath('.//td[6]/text()')[0],
        'Volume':row.xpath('.//td[7]/text()')[0]
    }) 
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
str_cols = ['High', 'Low', 'Close', 'Adj Close', 'Volume']
df[str_cols]=df[str_cols].replace(',', '', regex=True).astype(float)
df.dropna(inplace=True)
df = df.set_index('Date')
df.head()

sns.set_style('darkgrid')
plt.style.use('ggplot')
plt.figure(figsize=(15, 6))
df['Adj Close'].plot()
plt.ylabel('Adj Close')
plt.xlabel(None)
plt.title('Closing Price of AAPL')
plt.show()

features = ['Open', 'High', 'Low', 'Volume']
y = df.filter(['Adj Close'])
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])

tscv = TimeSeriesSplit(n_splits=10) 
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=8)

y_pred= model.predict(X_test)

plt.figure(figsize=(15, 6))
plt.plot(y_test.values, label='Actual Value')
plt.plot(y_pred, label='Predicted Value')
plt.ylabel('Adjusted Close (Scaled)')
plt.xlabel('Time Scale')
plt.legend()
plt.show()
