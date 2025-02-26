# ------------------- الإصدار المصحح مع التعليقات التوضيحية -------------------
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
import pytz
import json
import os
import traceback
# ------------------- إعدادات النظام -------------------
SAUDI_TZ = pytz.timezone('Asia/Riyadh')
PLANETS = ['sun', 'moon']
REPORT_DIR = 'النتائج'
MODEL_DIR = 'نماذج'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class FastAstroTrader:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.historical_data = self.load_stock_data()
        self.model = self.init_model()
        
    def load_stock_data(self):
        # إضافة تحقق من جودة البيانات
        data = yf.download(self.stock_symbol, start='2023-01-01')
        if data.empty:
            raise ValueError("فشل في تحميل البيانات، الرجاء التحقق من الرمز المالي")
        return data[['Close']]
    
    def init_model(self):
        return MLPClassifier(hidden_layer_sizes=(20,), max_iter=500)
    
    def quick_analyze(self):
        print("جاري التحليل السريع...")
        
        # الإصلاح الرئيسي هنا: تسطيح المصفوفة
        close_prices = self.historical_data['Close'].values.flatten()  # <-- التعديل هنا
        
        peaks, troughs = self.find_key_points(close_prices)
        print(f"تم تحديد {len(peaks)} قمم و {len(troughs)} قيعان")
        
        prediction = self.quick_predict()
        self.save_result(prediction)
        return prediction
    
    def find_key_points(self, close):
        # تأكيد أن البيانات 1D
        if close.ndim != 1:
            close = close.flatten()
            
        peaks, _ = find_peaks(close, prominence=np.std(close)/5)
        troughs, _ = find_peaks(-close, prominence=np.std(close)/5)
        return peaks, troughs
    
    def quick_predict(self):
        now = datetime.now(SAUDI_TZ)
        positions = self.get_planetary_positions(now)
        
        # تحويل القيم إلى مصفوفة numpy بشكل صحيح
        X = np.array([list(positions.values())]).reshape(1, -1)  # <-- التعديل هنا
        
        prediction = self.model.predict(X)[0]
        
        return {
            'تاريخ': now.strftime('%Y-%m-%d %H:%M'),
            'التنبؤ': 'صعود' if prediction == 1 else 'هبوط',
            'الكواكب_المؤثرة': self.get_active_planets(positions),
            'السعر_الحالي': self.historical_data['Close'].iloc[-1]
        }
    
    def get_planetary_positions(self, date):
        positions = {}
        t = self.ts.utc(date.year, date.month, date.day)
        
        for planet in PLANETS:
            body = self.eph[planet]
            astro = self.eph['earth'].at(t).observe(body)
            positions[planet] = astro.apparent().ecliptic_latlon()[1].degrees % 360
            
        return positions
    
    def get_active_planets(self, positions):
        return [p for p, angle in positions.items() if angle % 30 < 5]
    
    def save_result(self, result):
        with open(f'{REPORT_DIR}/نتيجة_سريعة.json', 'w') as f:
            json.dump(result, f, ensure_ascii=False)
        print("تم حفظ النتائج في ملف 'نتيجة_سريعة.json'")

if __name__ == "__main__":
    try:
        analyzer = FastAstroTrader('2222.SR')
        result = analyzer.quick_analyze()
        
        print("\nالنتائج الفورية:")
        print(f"التنبؤ: {result['التنبؤ']}")
        print(f"الكواكب النشطة: {', '.join(result['الكواكب_المؤثرة'])}")
        print(f"السعر الأخير: {result['السعر_الحالي']:.2f}")
    except Exception as e:
        print(f"حدث خطأ: {str(e)}")
        print("الرجاء التأكد من:")
        print("- اتصال الإنترنت يعمل")
        print("- الرمز المالي صحيح (مثال: 2220.SR)")
        print("- وجود مساحة كافية على القرص")
        traceback.print_exc()