# ------------------- الاستيرادات -------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, Topos
from sklearn.neural_network import MLPClassifier
import joblib
import pytz
import json
import os
import schedule
import time
from collections import Counter

# ------------------- الإعدادات -------------------
SAUDI_TZ = pytz.timezone('Asia/Riyadh')
PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars']
REPORT_DIR = 'النتائج'
MODEL_DIR = 'نماذج'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- الفئة الرئيسية -------------------
class AstroStockPredictor:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.historical_data = self._fetch_stock_data()
        self.model, self.training_data = self._load_or_init_model()
        self.predictions_log = self._load_predictions_log()

    # ------------------- الدوال الأساسية -------------------
    def _fetch_stock_data(self):
        """جلب البيانات مع معالجة التعديل التلقائي"""
        try:
            ticker = yf.Ticker(self.stock_symbol)
            start_date = pd.to_datetime(ticker.info.get('firstTradeDateMilliseconds', 0), 
                                      unit='ms', errors='coerce').strftime('%Y-%m-%d')
            data = yf.download(self.stock_symbol, start=start_date, 
                             progress=False, auto_adjust=True)
            return data[['Open', 'Close']] if not data.empty else pd.DataFrame()
        except Exception as e:
            print(f"فشل جلب بيانات {self.stock_symbol}: {str(e)}")
            return pd.DataFrame()

    def _calculate_planetary_positions(self, date):
        """حساب المواقع الكوكبية بدقة"""
        positions = {}
        t = self.ts.utc(date.year, date.month, date.day)
        earth = self.eph['earth']
        for body in PLANETS:
            planet = self.eph[body]
            astro = earth.at(t).observe(planet)
            _, ecliptic_lon, _ = astro.ecliptic_latlon()
            positions[body] = ecliptic_lon.degrees % 360
        return positions

    # ------------------- نظام التعلم الذاتي -------------------
    def daily_self_learning(self):
        """التحديث اليومي التلقائي"""
        try:
            # جلب بيانات اليوم
            new_data = yf.download(self.stock_symbol, period='1d', progress=False)
            if new_data.empty:
                return False

            # حساب القيم الفعلية
            actual = 1 if new_data['Close'].iloc[-1] > new_data['Open'].iloc[0] else 0
            
            # تحضير ميزات الأمس
            yesterday = datetime.now(SAUDI_TZ) - timedelta(days=1)
            features = self._prepare_features(yesterday)
            
            # تحديث بيانات التدريب
            self.training_data['X'].append(features.tolist())
            self.training_data['y'].append(actual)
            
            # تسجيل التنبؤات
            self._log_prediction(yesterday, features, actual)
            
            # إعادة التدريب الأسبوعي
            if datetime.now(SAUDI_TZ).weekday() == 3:  # كل خميس
                self._retrain_model()
            
            return True
        except Exception as e:
            print(f"فشل التعلم اليومي لـ {self.stock_symbol}: {str(e)}")
            return False

    def _log_prediction(self, date, features, actual):
        """تسجيل نتائج اليوم"""
        try:
            proba = self.model.predict_proba([features])[0]
            prediction = self.model.predict([features])[0]
            
            self.predictions_log['daily'].append({
                'date': date.strftime("%Y-%m-%d"),
                'prediction': int(prediction),
                'confidence': round(float(proba.max())*100, 2),
                'actual': int(actual),
                'correct': bool(prediction == actual)
            })
            
            with open(f'{REPORT_DIR}/{self.stock_symbol}_predictions.json', 'w', encoding='utf-8') as f:
                json.dump(self.predictions_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"فشل تسجيل النتائج: {str(e)}")

    def _retrain_model(self):
        """إعادة تدريب النموذج أسبوعيًا"""
        try:
            X = np.array(self.training_data['X'])
            y = np.array(self.training_data['y'])
            self.model.partial_fit(X, y)
            joblib.dump(self.model, f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
            print(f"تم إعادة تدريب النموذج لـ {self.stock_symbol}")
        except Exception as e:
            print(f"فشل إعادة التدريب: {str(e)}")

    # ------------------- التقارير التلقائية -------------------
    def generate_daily_report(self):
        """تقرير يومي تلقائي"""
        try:
            report = {
                'date': datetime.now(SAUDI_TZ).strftime("%Y-%m-%d"),
                'predictions': self.predict_extremes(days_ahead=1),
                'performance_stats': self._daily_performance()
            }
            
            with open(f'{REPORT_DIR}/{self.stock_symbol}_daily_{datetime.now().date()}.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            return report
        except Exception as e:
            print(f"فشل توليد التقرير اليومي: {str(e)}")
            return {}

    def generate_weekly_report(self):
        """تقرير أسبوعي تلقائي"""
        try:
            weekly_data = self.predictions_log['daily'][-7:]
            
            report = {
                'week_start': weekly_data[0]['date'],
                'week_end': weekly_data[-1]['date'],
                'accuracy': self._calculate_accuracy(weekly_data),
                'avg_confidence': np.mean([d['confidence'] for d in weekly_data]),
                'recommendations': self._generate_recommendations()
            }
            
            with open(f'{REPORT_DIR}/{self.stock_symbol}_weekly_{datetime.now().date()}.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            return report
        except Exception as e:
            print(f"فشل توليد التقرير الأسبوعي: {str(e)}")
            return {}

    # ------------------- الدوال المساعدة -------------------
    def _load_predictions_log(self):
        """تحميل سجل التنبؤات"""
        try:
            with open(f'{REPORT_DIR}/{self.stock_symbol}_predictions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'daily': [], 'weekly': []}

    def _daily_performance(self):
        """أداء اليوم"""
        today_log = [d for d in self.predictions_log['daily'] 
                    if d['date'] == datetime.now().strftime("%Y-%m-%d")]
        return {
            'total_predictions': len(today_log),
            'accuracy': sum(1 for d in today_log if d['correct'])/len(today_log) if today_log else 0
        }

    def _calculate_accuracy(self, data):
        """حساب الدقة"""
        correct = sum(1 for d in data if d['correct'])
        return round(correct/len(data)*100, 2) if data else 0.0

    def _generate_recommendations(self):
        """توصيات التحسين"""
        aspects = Counter([aspect for d in self.predictions_log['daily'][-7:] 
                         for aspect in d.get('aspects', [])])
        return [f"زيادة وزن {k}" for k, v in aspects.most_common(3)]

# ------------------- الجدولة التلقائية -------------------
def schedule_tasks():
    """جدولة المهام اليومية والأسبوعية"""
    # مهمة يومية الساعة 4 مساءً
    schedule.every().day.at("21:46", SAUDI_TZ).do(run_daily_tasks)
    
    # مهمة أسبوعية مساء الخميس
    schedule.every().wednesday.at("21:46", SAUDI_TZ).do(run_weekly_tasks)

def run_daily_tasks():
    """المهام اليومية لجميع الأسهم"""
    stocks = ["2220.SR", "1111.SR", "1120.SR", "1211.SR", "2380.SR"]
    for symbol in stocks:
        try:
            predictor = AstroStockPredictor(symbol)
            predictor.daily_self_learning()
            predictor.generate_daily_report()
            print(f"تمت معالجة {symbol} اليومية بنجاح")
        except Exception as e:
            print(f"فشل المعالجة اليومية لـ {symbol}: {str(e)}")

def run_weekly_tasks():
    """المهام الأسبوعية لجميع الأسهم"""
    stocks = ["2220.SR", "1111.SR", "1120.SR", "1211.SR", "2380.SR"]
    for symbol in stocks:
        try:
            predictor = AstroStockPredictor(symbol)
            predictor.generate_weekly_report()
            print(f"تمت معالجة {symbol} الأسبوعية بنجاح")
        except Exception as e:
            print(f"فشل المعالجة الأسبوعية لـ {symbol}: {str(e)}")

# ------------------- التشغيل الرئيسي -------------------
if __name__ == "__main__":
    print("بدء تشغيل نظام التداول الذاتي...")
    schedule_tasks()
    
    while True:
        schedule.run_pending()
        time.sleep(60)