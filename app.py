import yfinance as yf
import pandas as pd
from datetime import datetime
from skyfield.api import load
from scipy.stats import chi2_contingency
from tqdm import tqdm

# تهيئة البيانات الفلكية
ts = load.timescale()
planets = load('de421.bsp')

# قائمة الكواكب المدعومة
all_bodies = ['sun', 'moon', 'mercury', 'venus', 'mars', 'earth']

# دالة لحساب تاريخ الإدراج
def get_listing_date(stock):
    ticker = yf.Ticker(stock)
    history = ticker.history(period='max')
    if history.empty:
        print(f"No data available for {stock}.")
        return None
    return history.index[0].date()

# دالة لحساب مواقع الكواكب
def get_planet_positions(date):
    t = ts.utc(date.year, date.month, date.day)
    positions = {}
    for body in all_bodies:
        astrometric = planets[body].at(t).observe(planets['earth']).apparent()
        ra, dec, _ = astrometric.radec()
        positions[body] = ra.degrees % 360
    return positions

# تحديث قائمة الأسهم
stocks = [
    '2220.SR', '8110.SR', '2030.SR', '1120.SR', 
    '1010.SR', '3010.SR', '7010.SR', '2080.SR', 
    '4040.SR', '4110.SR',
]

results = []

# قائمة الزوايا الديناميكية
angles = range(0, 360, 10)  # زوايا من 0 إلى 360 على فترات 10 درجات

for stock in tqdm(stocks):
    listing_date = get_listing_date(stock)
    if listing_date is None:
        continue

    data = yf.download(stock, start=listing_date, end=datetime.now().date())
    data = data[['Close']].reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['positions'] = data['Date'].apply(get_planet_positions)

    for body in all_bodies:
        data[f'{body}_ra'] = data['positions'].apply(lambda x: x[body])

    data['peak'] = data['Close'].rolling(5).apply(lambda x: 1 if x[-1] == x.max() else 0, raw=True)

    for angle in angles:
        for body in all_bodies:
            col = f'{body}_ra'
            in_range = data[col].between(angle - 10, angle + 10, inclusive='both')
            
            peak_prob = data[in_range]['peak'].mean() if in_range.any() else 0
            
            if in_range.any():
                contingency = pd.crosstab(in_range, data['peak'])
                
                if contingency.size > 0:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    results.append({
                        'stock': stock,
                        'planet': body,
                        'angle': angle,
                        'peak_prob': peak_prob,
                        'p_value': p,
                        'count': in_range.sum()
                    })
                else:
                    print(f"No data in contingency for {body} at angle {angle}.")
            else:
                print(f"No days in range for {body} at angle {angle}.")

# تحويل النتائج إلى DataFrame
results_df = pd.DataFrame(results)

# النتائج ذات الدلالة الإحصائية (p < 0.05) وتكرار ≥ 10
significant_results = results_df[
    (results_df['p_value'] < 0.05) & 
    (results_df['count'] >= 10)
].sort_values('peak_prob', ascending=False)

print("الأنماط الأكثر تأثيرًا:")
print(significant_results[['stock', 'planet', 'angle', 'peak_prob']])