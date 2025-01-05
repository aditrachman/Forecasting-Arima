import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Load dataset
file_path = 'data_resik.csv'
dataset = pd.read_csv(file_path)
dataset['votes'] = dataset['votes'].str.replace(',', '').astype(int)
trend_data = dataset[['year', 'genre']]
genre_trend = trend_data.groupby(['year', 'genre']).size().reset_index(name='count')

# Filter data untuk genre Horror
horror_trend = genre_trend[genre_trend['genre'].str.contains('Horror', case=False)]
warnings.filterwarnings('ignore')
horror_ts = horror_trend.set_index('year')['count']
model = ARIMA(horror_ts, order=(2, 1, 2)) 
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
forecast_years = list(range(horror_ts.index.max() + 1, horror_ts.index.max() + 11))

#ploting forecasting
plt.figure(figsize=(12, 6))
plt.plot(horror_ts, label='Actual Data', color='red', marker='o')
plt.plot(forecast_years, forecast, label='Forecast', color='blue', linestyle='--', marker='x')
plt.title('Peramalan Tren Film Horror', fontsize=16)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Jumlah Film', fontsize=12)
plt.legend()
plt.grid(alpha=0.4)
plt.show()
