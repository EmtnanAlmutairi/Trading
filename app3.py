import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, Topos
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
import joblib
import schedule
import time
import pytz
import json
import os
import ephem

# ------------------- إعدادات النظام -------------------
SAUDI_TZ = pytz.timezone('Asia/Riyadh')
PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars', 
          'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
REPORT_DIR = 'النتائج'
MODEL_DIR = 'نماذج'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- الفئة الأساسية للكشف عن الأنماط الفلكية -------------------
class AstroPatternDetector:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.historical_data = self.load_stock_data()
        self.planetary_cycles = self.calculate_planetary_cycles()

    def load_stock_data(self):
        data = yf.download(self.stock_symbol, start='2000-01-01')
        return data[['Close']]

    def calculate_planetary_positions(self, date):
        positions = {}
        t = self.ts.utc(date.year, date.month, date.day)
        earth = self.eph['earth']
        
        for planet in PLANETS:
            body = self.eph[planet]
            astro = earth.at(t).observe(body)
            ecliptic_lat, ecliptic_lon = astro.ecliptic_latlon()
            positions[planet] = ecliptic_lon.degrees % 360
            
        return positions

    def detect_price_extremes(self, window=30):
        close = self.historical_data['Close'].values
        peaks, _ = find_peaks(close, prominence=np.std(close)/3)
        troughs, _ = find_peaks(-close, prominence=np.std(close)/3)
        return peaks, troughs

    def map_astro_patterns(self, extremes):
        patterns = []
        for idx in extremes:
            date = self.historical_data.index[idx]
            price = self.historical_data.iloc[idx]['Close']
            positions = self.calculate_planetary_positions(date)
            
            pattern = {
                'date': date,
                'price_level': price,
                'angles': positions
            }
            patterns.append(pattern)
        return patterns

    def cluster_patterns(self, patterns, eps=15):
        angles_matrix = np.array([[v for v in p['angles'].values()] for p in patterns])
        clustering = DBSCAN(eps=eps, min_samples=3).fit(angles_matrix)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(patterns[i])
        return clusters

    def calculate_planetary_cycles(self):
        return {
            'mercury': 88, 'venus': 225, 'mars': 687,
            'jupiter': 4333, 'saturn': 10759,
            'uranus': 30685, 'neptune': 60190, 'pluto': 90520
        }

    def predict_future_extremes(self, clusters, years=5):
        predictions = []
        today = datetime.now()
        
        for cluster_id, patterns in clusters.items():
            if cluster_id == -1 or len(patterns) < 3:
                continue
            
            dates = [p['date'] for p in patterns]
            time_diffs = np.diff([d.timestamp() for d in dates]).mean()
            
            last_date = max(dates)
            cycle_days = int((time_diffs / (60*60*24)) * 1.1)
            
            current_date = last_date
            while (current_date - today).days < years*365:
                current_date += timedelta(days=cycle_days)
                predictions.append({
                    'date': current_date,
                    'cluster': cluster_id,
                    'confidence': min(90, len(patterns)*15)
                })
                
        return predictions

    def analyze_planet_cycle(self, planet):
        significant_angles = []
        historical_angles = []
        
        for date in self.historical_data.index:
            pos = self.calculate_planetary_positions(date)[planet]
            historical_angles.append(pos)
        
        fft = np.fft.fft(historical_angles)
        freqs = np.fft.fftfreq(len(historical_angles))
        dominant_freq = freqs[np.argmax(np.abs(fft))]
        
        angle_changes = np.diff(historical_angles)
        threshold = np.std(angle_changes) * 1.5
        for i in range(1, len(historical_angles)):
            if abs(angle_changes[i-1]) > threshold:
                significant_angles.append({
                    'date': self.historical_data.index[i],
                    'angle': historical_angles[i],
                    'price_impact': self.calculate_price_impact(i)
                })
        
        return significant_angles

    def calculate_price_impact(self, idx):
        window = 30
        start = max(0, idx - window)
        end = min(len(self.historical_data), idx + window)
        price_window = self.historical_data.iloc[start:end]['Close']
        return (price_window.max() - price_window.min()) / price_window.mean()

# ------------------- فئة التعلم الذاتي اليومي -------------------
class SelfLearningAstroTrader(AstroPatternDetector):
    def __init__(self, stock_symbol):
        super().__init__(stock_symbol)
        self.model = self.load_latest_model()
        self.today = datetime.now(SAUDI_TZ)
        
    def load_latest_model(self):
        try:
            return joblib.load(f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
        except:
            return MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
            
    def daily_data_pipeline(self):
        new_data = yf.download(self.stock_symbol, period='1d')
        positions = self.calculate_planetary_positions(self.today)
        aspects = self.detect_aspects(positions)
        X = self.prepare_astro_features(positions, aspects)
        y = self.calculate_price_movement(new_data['Close'])
        return X, y
    
    def prepare_astro_features(self, positions, aspects):
        features = list(positions.values())
        
        aspect_types = {'اقتران':0, 'تربيع':0, 'تقابل':0}
        for aspect in aspects:
            key = aspect.split()[0]
            aspect_types[key] += 1
        features += list(aspect_types.values())
        
        features.append(self.is_retrograde(self.today))
        
        return np.array(features).reshape(1, -1)
    
    def is_retrograde(self, date):
        t = self.ts.utc(date.year, date.month, date.day)
        mercury = self.eph['mercury']
        earth = self.eph['earth']
        velocity = earth.at(t).observe(mercury).apparent().velocity.km_per_s
        return 1 if velocity < 0 else 0
    
    def calculate_price_movement(self, prices):
        open_price = prices.iloc[0]
        close_price = prices.iloc[-1]
        return 1 if close_price > open_price else 0
    
    def update_model(self, X, y):
        try:
            history = joblib.load(f'{MODEL_DIR}/{self.stock_symbol}_training_data.pkl')
            X_hist, y_hist = history['X'], history['y']
            X = np.vstack([X_hist, X])
            y = np.concatenate([y_hist, y])
        except:
            pass
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.partial_fit(X_train, y_train, classes=[0, 1])
        
        joblib.dump({'X':X, 'y':y}, f'{MODEL_DIR}/{self.stock_symbol}_training_data.pkl')
        joblib.dump(self.model, f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
        
        accuracy = self.model.score(X_test, y_test)
        print(f'دقة النموذج المحدث: {accuracy:.2f}')
        
    def generate_tomorrow_prediction(self):
        tomorrow = self.today + timedelta(days=1)
        positions = self.calculate_planetary_positions(tomorrow)
        aspects = self.detect_aspects(positions)
        X = self.prepare_astro_features(positions, aspects)
        
        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)
        
        return {
            'date': tomorrow.strftime('%Y-%m-%d'),
            'prediction': 'صعود' if prediction[0] == 1 else 'هبوط',
            'confidence': np.max(probability)*100,
            'critical_angles': self.get_critical_angles(positions)
        }
    
    def get_critical_angles(self, positions):
        thresholds = {
            'sun': 5, 'moon':3, 'mercury':7,
            'venus':4, 'mars':6, 'jupiter':9
        }
        
        critical = []
        for planet, angle in positions.items():
            if planet in thresholds:
                if (angle % thresholds[planet]) < 1:
                    critical.append(f'{planet}: {angle:.1f}°')
        return critical
        
    def daily_learning_cycle(self):
        print(f'بدء التعلم الذاتي: {self.today}')
        
        try:
            X, y = self.daily_data_pipeline()
            self.update_model(X, y)
            prediction = self.generate_tomorrow_prediction()
            self.save_daily_report(prediction)
            print('اكتملت دورة التعلم بنجاح!')
        except Exception as e:
            print(f'خطأ في دورة التعلم: {str(e)}')
    
    def save_daily_report(self, prediction):
        report = {
            'date': datetime.now().isoformat(),
            'prediction': prediction,
            'model_version': joblib.hash(self.model),
            'training_samples': len(self.model.classes_)
        }
        
        with open(f'{REPORT_DIR}/daily_log.json', 'a') as f:
            f.write(json.dumps(report, ensure_ascii=False) + '\n')

# ------------------- تشغيل النظام -------------------
def run_daily_task():
    for stock in ['2220.SR', '1111.SR']:
        trader = SelfLearningAstroTrader(stock)
        trader.daily_learning_cycle()

schedule.every().day.at("16:00", SAUDI_TZ).do(run_daily_task)

if __name__ == "__main__":
    print('نظام التحليل الفلكي الذاتي التشغيل...')
    while True:
        schedule.run_pending()
        time.sleep(60)