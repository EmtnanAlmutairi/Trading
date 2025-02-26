# ------------------- الاستيرادات -------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, Topos
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import pytz
import json
import os
from collections import Counter

# ------------------- الإعدادات -------------------
SAUDI_TZ = pytz.timezone('Asia/Riyadh')
PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars']
REPORT_DIR = 'النتائج'
MODEL_DIR = 'نماذج'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- الفئة الأساسية -------------------
class AstroStockPredictor:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.historical_data = self._fetch_stock_data()
        self.model, self.training_data = self._load_or_init_model()

    def _fetch_stock_data(self):
        """جلب البيانات التاريخية مع التعامل مع التعديل التلقائي"""
        try:
            ticker = yf.Ticker(self.stock_symbol)
            start_date = pd.to_datetime(ticker.info.get('firstTradeDateMilliseconds', 0), unit='ms', errors='coerce').strftime('%Y-%m-%d')
            data = yf.download(self.stock_symbol, start=start_date, progress=False, auto_adjust=True)
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

    def _detect_aspects(self, positions):
        """كشف الجوانب الفلكية مع تحسين الدقة"""
        aspects = []
        aspect_config = {0: 8, 60: 5, 90: 6, 120: 5, 180: 8}
        planets = list(positions.keys())
        for i in range(len(planets)):
            for j in range(i+1, len(planets)):
                angle = abs(positions[planets[i]] - positions[planets[j]]) % 360
                angle = min(angle, 360 - angle)
                for aspect_angle, orb in aspect_config.items():
                    if abs(angle - aspect_angle) <= orb:
                        aspects.append(aspect_angle)
                        break
        return aspects

    def _prepare_features(self, date):
        """إعداد الميزات مع 11 خاصية"""
        positions = self._calculate_planetary_positions(date)
        aspects = self._detect_aspects(positions)
        
        features = list(positions.values())  # 5 ميزات
        aspect_counts = Counter(aspects)
        features += [aspect_counts.get(a, 0) for a in [0, 60, 90, 120, 180]]  # +5
        features.append(self._check_retrograde(date))  # +1
        return np.array(features)

    def _check_retrograde(self, date):
        """فحص تراجع عطارد بدقة أعلى"""
        try:
            t = self.ts.utc(date.year, date.month, date.day)
            mercury = self.eph['mercury']
            earth = self.eph['earth']
            astro = earth.at(t).observe(mercury)
            velocity = astro.apparent().velocity.km_per_s
            return 1 if velocity[0] < 0 else 0
        except Exception as e:
            print(f"خطأ في فحص التراجع: {str(e)}")
            return 0

    def _load_or_init_model(self):
        """تهيئة النموذج مع تحميل ذكي"""
        model_path = f'{MODEL_DIR}/{self.stock_symbol}_model.pkl'
        data_path = f'{MODEL_DIR}/{self.stock_symbol}_data.pkl'
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            return joblib.load(model_path), joblib.load(data_path)
        else:
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=3000, early_stopping=True)
            training_data = {'X': [], 'y': []}
            self._train_initial_model(model, training_data)
            return model, training_data

    def _train_initial_model(self, model, training_data):
        """تدريب النموذج مع معالجة الأخطاء"""
        X, y = [], []
        for i in range(1, len(self.historical_data)):
            try:
                date = self.historical_data.index[i]
                features = self._prepare_features(date - timedelta(days=1))
                close_val = self.historical_data.iloc[i]['Close']
                open_val = self.historical_data.iloc[i]['Open']
                price_move = 1 if close_val > open_val else 0
                X.append(features)
                y.append(price_move)
            except Exception as e:
                print(f"خطأ في التدريب: {str(e)}")
                continue
        
        if len(X) > 100:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model.fit(X_train, y_train)
            training_data.update({'X': X, 'y': y})
            joblib.dump(model, f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
            joblib.dump(training_data, f'{MODEL_DIR}/{self.stock_symbol}_data.pkl')

    def predict_yearly_extremes(self):
        """التنبؤ بأهم قمة وقاع للسنة الحالية"""
        today = datetime.now(SAUDI_TZ)
        end_of_year = datetime(today.year, 12, 31, tzinfo=SAUDI_TZ)
        days_remaining = (end_of_year - today).days
        
        peaks = []
        troughs = []
        
        for i in range(days_remaining):
            current_date = today + timedelta(days=i)
            try:
                features = self._prepare_features(current_date)
                proba = self.model.predict_proba([features])[0]
                confidence = round(max(proba)*100, 2)
                prediction_type = 'peak' if np.argmax(proba) == 1 else 'trough'
                
                # حساب الأهمية الفلكية
                positions = self._calculate_planetary_positions(current_date)
                significance = self._calculate_significance(positions)
                
                if significance < 40:  # تجاهل الأحداث غير المهمة
                    continue
                
                # جمع النتائج
                if prediction_type == 'peak' and proba[1] > 0.65:
                    peaks.append({
                        'date': current_date.strftime("%Y-%m-%d"),
                        'confidence': confidence,
                        'significance': significance,
                        'aspects': self._get_aspect_details(current_date),
                        'critical_planets': self._get_critical_planets(positions)
                    })
                elif prediction_type == 'trough' and proba[0] > 0.6:
                    troughs.append({
                        'date': current_date.strftime("%Y-%m-%d"),
                        'confidence': confidence,
                        'significance': significance,
                        'aspects': self._get_aspect_details(current_date),
                        'critical_planets': self._get_critical_planets(positions)
                    })
                    
            except Exception as e:
                print(f"خطأ في تاريخ {current_date}: {str(e)}")
                continue
        
        # اختيار الأقوى
        best_peak = max(peaks, key=lambda x: x['confidence'], default=None)
        best_trough = max(troughs, key=lambda x: x['confidence'], default=None)
        
        return {
            'best_peak': best_peak,
            'best_trough': best_trough,
            'analyzed_days': days_remaining,
            'total_peaks': len(peaks),
            'total_troughs': len(troughs)
        }

    def _calculate_significance(self, positions):
        """حساب قوة الإشارة الفلكية"""
        critical_count = 0
        for pos in positions.values():
            remainder = pos % 30
            if remainder < 2 or (30 - remainder) < 2:
                critical_count += 1
        return critical_count * 25  # 25% لكل زاوية حرجة

    def _get_critical_planets(self, positions):
        """الكواكب في مواقع حرجة"""
        critical = []
        for planet, pos in positions.items():
            remainder = pos % 30
            if remainder < 2 or (30 - remainder) < 2:
                critical.append({
                    'planet': planet,
                    'angle': round(pos, 1),
                    'interpretation': self._get_angle_interpretation(pos)
                })
        return critical

    def _get_angle_interpretation(self, angle):
        """تفسير الزوايا الحرجة"""
        remainder = angle % 30
        if remainder < 2:
            return "بداية برج جديد"
        elif (30 - remainder) < 2:
            return "نهاية برج"
        return ""

    def _get_aspect_details(self, date):
        """تفاصيل الجوانب الفلكية"""
        positions = self._calculate_planetary_positions(date)
        aspects = []
        planets = list(positions.keys())
        for i in range(len(planets)):
            for j in range(i+1, len(planets)):
                angle = abs(positions[planets[i]] - positions[planets[j]]) % 360
                angle = min(angle, 360 - angle)
                aspect_type = None
                if angle <= 8:
                    aspect_type = "اقتران"
                elif 55 <= angle <= 65:
                    aspect_type = "سداسي"
                elif 85 <= angle <= 95:
                    aspect_type = "تربيع"
                elif 115 <= angle <= 125:
                    aspect_type = "ثلاثي"
                elif 175 <= angle <= 185:
                    aspect_type = "تقابل"
                
                if aspect_type:
                    aspects.append(f"{aspect_type} بين {planets[i]} و{planets[j]}")
        return aspects

    def generate_yearly_report(self):
        """توليد تقرير سنوي مفصل"""
        if self.historical_data.empty or len(self.historical_data) < 200:
            print(f"بيانات غير كافية لـ {self.stock_symbol}")
            return None
        
        extremes = self.predict_yearly_extremes()
        
        report = {
            'stock': self.stock_symbol,
            'year': datetime.now().year,
            'report_date': datetime.now(SAUDI_TZ).strftime("%Y-%m-%d"),
            'predictions': {
                'most_likely_peak': extremes['best_peak'],
                'most_likely_trough': extremes['best_trough']
            },
            'analysis_metrics': {
                'historical_accuracy': self._calculate_accuracy(),
                'days_analyzed': extremes['analyzed_days'],
                'total_signals': extremes['total_peaks'] + extremes['total_troughs']
            }
        }
        
        # حفظ التقرير
        report_path = f"{REPORT_DIR}/{self.stock_symbol}_yearly_{datetime.now().year}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
        return report

    def _calculate_accuracy(self):
        """حساب دقة النموذج بدقة أعلى"""
        try:
            if len(self.historical_data) < 200:
                return 0.0
                
            correct = 0
            test_days = len(self.historical_data) - 200
            
            for i in range(200, len(self.historical_data)):
                try:
                    date = self.historical_data.index[i]
                    features = self._prepare_features(date - timedelta(days=1))
                    prediction = self.model.predict([features])[0]
                    actual = 1 if self.historical_data.iloc[i]['Close'] > self.historical_data.iloc[i]['Open'] else 0
                    if prediction == actual:
                        correct += 1
                except:
                    continue
                    
            return round((correct / test_days) * 100, 2) if test_days > 0 else 0.0
        except Exception as e:
            print(f"خطأ في حساب الدقة: {str(e)}")
            return 0.0

# ------------------- التشغيل الرئيسي -------------------
if __name__ == "__main__":
    # قائمة الأسهم السعودية المختارة
    selected_stocks = [
        "2220.SR",  # أرامكو
        "1111.SR",  # السعودية للكهرباء
        "1120.SR",  # الراجحي
        "1211.SR",  # الأهلي
        "4150.SR",  # اتصالات
        "2380.SR",  # بيت الراجحي
        "4001.SR",  # الأسمنت
        "6001.SR"   # الرياض
    ]
    
    for symbol in selected_stocks:
        try:
            print(f"\n{'='*50}")
            print(f"بدء تحليل {symbol} للسنة {datetime.now().year}")
            
            # إنشاء المحلل
            analyzer = AstroStockPredictor(symbol)
            
            # التحقق من جودة البيانات
            if analyzer.historical_data.empty:
                print(f" - فشل: لا توجد بيانات تاريخية")
                continue
                
            if len(analyzer.historical_data) < 365:
                print(f" - تحذير: بيانات أقل من سنة (أيام التداول: {len(analyzer.historical_data)})")
            
            # توليد التقرير
            report = analyzer.generate_yearly_report()
            
            if not report:
                continue
                
            # عرض النتائج
            print("\nالنتائج الرئيسية:")
            if report['predictions']['most_likely_peak']:
                peak = report['predictions']['most_likely_peak']
                print(f"\nأقوى توقع قمة:")
                print(f" - التاريخ: {peak['date']}")
                print(f" - الثقة: {peak['confidence']}%")
                print(f" - الأهمية الفلكية: {peak['significance']}%")
                print(f" - الكواكب الحرجة:")
                for planet in peak['critical_planets']:
                    print(f"   • {planet['planet']}: {planet['angle']}° ({planet['interpretation']})")
                print(f" - الجوانب الفلكية: {', '.join(peak['aspects'])}")
                
            if report['predictions']['most_likely_trough']:
                trough = report['predictions']['most_likely_trough']
                print(f"\nأقوى توقع قاع:")
                print(f" - التاريخ: {trough['date']}")
                print(f" - الثقة: {trough['confidence']}%")
                print(f" - الأهمية الفلكية: {trough['significance']}%")
                print(f" - الكواكب الحرجة:")
                for planet in trough['critical_planets']:
                    print(f"   • {planet['planet']}: {planet['angle']}° ({planet['interpretation']})")
                print(f" - الجوانب الفلكية: {', '.join(trough['aspects'])}")
            
            print(f"\nمقاييس التحليل:")
            print(f" - الدقة التاريخية: {report['analysis_metrics']['historical_accuracy']}%")
            print(f" - عدد الأيام التي تم تحليلها: {report['analysis_metrics']['days_analyzed']}")
            print(f" - عدد الإشارات المكتشفة: {report['analysis_metrics']['total_signals']}")
            
        except Exception as e:
            print(f"! خطأ جسيم في معالجة {symbol}: {str(e)}")

    print("\nاكتملت عملية التحليل لجميع الأسهم المختارة")
# # ------------------- الاستيرادات -------------------
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from skyfield.api import load, Topos
# from sklearn.cluster import DBSCAN
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from scipy.signal import find_peaks
# import joblib
# import schedule
# import time
# import pytz
# import json
# import os
# from collections import Counter
# import traceback

# # ------------------- الإعدادات -------------------
# SAUDI_TZ = pytz.timezone('Asia/Riyadh')
# PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars']
# REPORT_DIR = 'النتائج'
# MODEL_DIR = 'نماذج'
# os.makedirs(REPORT_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)

# # ------------------- فئة التحليل الفلكي الأساسية -------------------
# class AstroPatternDetector:
#     def __init__(self, stock_symbol):
#         self.stock_symbol = stock_symbol
#         self.ts = load.timescale()
#         self.eph = load('de421.bsp')
#         self.historical_data = self.load_stock_data()
#         self.planetary_cycles = self.get_planetary_cycles()

#     def get_listing_date(self):
#         try:
#             ticker = yf.Ticker(self.stock_symbol)
#             info = ticker.info
#             if 'firstTradeDateMilliseconds' in info:
#                 timestamp = info['firstTradeDateMilliseconds'] / 1000
#                 return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
#             return '2000-01-01'
#         except:
#             return '2000-01-01'

#     def load_stock_data(self):
#         start_date = self.get_listing_date()
#         data = yf.download(self.stock_symbol, start=start_date, progress=False)
        
#         if data.empty or 'Close' not in data.columns or 'Open' not in data.columns:
#             raise ValueError(f"بيانات غير صالحة لـ {self.stock_symbol}")
            
#         return data[['Open', 'Close']]

#     def get_planetary_cycles(self):
#         return {'mercury': 88, 'venus': 225, 'mars': 687}

#     def calculate_planetary_positions(self, date):
#         positions = {}
#         t = self.ts.utc(date.year, date.month, date.day)
#         earth = self.eph['earth']
#         for body in PLANETS:
#             planet = self.eph[body]
#             astro = earth.at(t).observe(planet)
#             _, ecliptic_lon, _ = astro.ecliptic_latlon()
#             positions[body] = ecliptic_lon.degrees % 360
#         return positions

#     def detect_price_extremes(self, prominence=0.1):
#         close = self.historical_data['Close']
#         peaks, _ = find_peaks(close, prominence=prominence)
#         troughs, _ = find_peaks(-close, prominence=prominence)
#         return peaks, troughs

# # ------------------- فئة التداول الذكي -------------------
# class SelfLearningAstroTrader(AstroPatternDetector):
#     def __init__(self, stock_symbol):
#         super().__init__(stock_symbol)
#         self.today = datetime.now(SAUDI_TZ)
#         self.model, self.training_data = self.load_or_init_model()

#     def load_or_init_model(self):
#         try:
#             model = joblib.load(f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
#             training_data = joblib.load(f'{MODEL_DIR}/{self.stock_symbol}_data.pkl')
#             return model, training_data
#         except:
#             model = MLPClassifier(
#                 hidden_layer_sizes=(50, 30),
#                 max_iter=1000,
#                 early_stopping=True,
#                 validation_fraction=0.2
#             )
#             training_data = {'X': [], 'y': [], 'last_updated': self.today.isoformat()}
#             self.initialize_historical_training(model, training_data)
#             return model, training_data

#     def initialize_historical_training(self, model, training_data):
#         X, y = self.generate_historical_features()
#         if len(X) > 0:
#             model.fit(X, y)
#             training_data['X'] = X.tolist()
#             training_data['y'] = y.tolist()
#             self.save_model(model, training_data)

#     def generate_historical_features(self):
#         X, y = [], []
#         for i in range(1, len(self.historical_data)):
#             try:
#                 date = self.historical_data.index[i]
#                 prev_day = date - timedelta(days=1)
#                 positions = self.calculate_planetary_positions(prev_day)
#                 aspects = self.detect_aspects(positions)
#                 features = self.prepare_features(positions, aspects)
                
#                 close_val = self.historical_data.iloc[i]['Close'].item()
#                 open_val = self.historical_data.iloc[i]['Open'].item()
#                 price_move = 1 if close_val > open_val else 0
#                 X.append(features)
#                 y.append(price_move)
#             except Exception as e:
#                 print(f"Error processing {date}: {str(e)}")
#         return np.array(X), np.array(y)

#     def detect_aspects(self, positions, orb=5):
#         aspects = []
#         planets = list(positions.keys())
#         aspect_config = {0: 8, 60: 5, 90: 6, 120: 5, 180: 8}
#         for i in range(len(planets)):
#             for j in range(i+1, len(planets)):
#                 angle = abs(positions[planets[i]] - positions[planets[j]]) % 360
#                 angle = min(angle, 360 - angle)
#                 for aspect_angle, aspect_orb in aspect_config.items():
#                     if abs(angle - aspect_angle) <= aspect_orb:
#                         aspects.append(aspect_angle)
#                         break
#         return aspects

#     def prepare_features(self, positions, aspects):
#         features = list(positions.values())
#         aspect_counts = Counter(aspects)
#         features += [
#             aspect_counts.get(0, 0),
#             aspect_counts.get(60, 0),
#             aspect_counts.get(90, 0),
#             aspect_counts.get(120, 0),
#             aspect_counts.get(180, 0),
#             self.check_retrograde(self.today, 'mercury'),
#             self.check_retrograde(self.today, 'venus')
#         ]
#         return np.array(features)

#     def check_retrograde(self, date, planet):
#         try:
#             t = self.ts.utc(date.year, date.month, date.day)
#             planet_obj = self.eph[planet]
#             earth = self.eph['earth']
#             velocity = earth.at(t).observe(planet_obj).apparent().velocity.km_per_s
#             return 1 if velocity[0] < 0 else 0
#         except:
#             return 0

#     def daily_update(self):
#         try:
#             new_data = yf.download(self.stock_symbol, period='1d', progress=False)
#             if new_data.empty or 'Close' not in new_data.columns or 'Open' not in new_data.columns:
#                 print(f"بيانات غير كاملة لـ {self.stock_symbol}")
#                 return

#             close_value = new_data['Close'].iloc[-1].item()
#             open_value = new_data['Open'].iloc[0].item()
#             y_new = 1 if close_value > open_value else 0
            
#             yesterday = self.today - timedelta(days=1)
#             positions = self.calculate_planetary_positions(yesterday)
#             aspects = self.detect_aspects(positions)
#             X_new = self.prepare_features(positions, aspects)
            
#             self.training_data['X'].append(X_new.tolist())
#             self.training_data['y'].append(y_new)
#             self.training_data['last_updated'] = self.today.isoformat()
            
#             if len(self.training_data['X']) > 10:
#                 X = np.array(self.training_data['X'])
#                 y = np.array(self.training_data['y'])
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#                 self.model.fit(X_train, y_train)
            
#             self.save_model(self.model, self.training_data)
#         except Exception as e:
#             print(f"فشل التحديث اليومي لـ {self.stock_symbol}: {str(e)}")

#     def save_model(self, model, data):
#         joblib.dump(model, f'{MODEL_DIR}/{self.stock_symbol}_model.pkl')
#         joblib.dump(data, f'{MODEL_DIR}/{self.stock_symbol}_data.pkl')

# # ------------------- فئة التقارير المتقدمة -------------------
# class AdvancedStockReporter(SelfLearningAstroTrader):
#     def __init__(self, stock_symbol):
#         super().__init__(stock_symbol)
#         self.historical_reports = self.load_history()

#     def load_history(self):
#         try:
#             with open(f'{REPORT_DIR}/{self.stock_symbol}_reports.json', 'r') as f:
#                 return json.load(f)
#         except:
#             return {'reports': [], 'performance_stats': {}}

#     def calculate_accuracy(self):
#         correct = 0
#         total = 0
#         for report in self.historical_reports['reports']:
#             for pred in report['predictions']:
#                 actual = self.get_actual_movement(pred['date'])
#                 if actual is None:
#                     continue
                
#                 if (pred['type'] == 'peak' and actual == 1) or (pred['type'] == 'trough' and actual == 0):
#                     correct += 1
#                 total += 1
#         return round(correct/total*100, 2) if total > 0 else 0.0

#     def get_actual_movement(self, target_date):
#         try:
#             target = pd.to_datetime(target_date).date()
#             data = self.historical_data[self.historical_data.index.date == target]
#             if not data.empty:
#                 return 1 if data['Close'].iloc[0].item() > data['Open'].iloc[0].item() else 0
#             return None
#         except:
#             return None

#     def generate_full_report(self):
#         try:
#             report = {
#                 'symbol': self.stock_symbol,
#                 'report_date': self.today.isoformat(),
#                 'listing_date': self.get_listing_date(),
#                 'predictions': self.predict_extremes(),
#                 'model_performance': {
#                     'accuracy': self.calculate_accuracy(),
#                     'training_days': len(self.training_data['X']),
#                     'last_update': self.training_data.get('last_updated', 'N/A')
#                 },
#                 'price_history_stats': self.get_price_stats()
#             }
            
#             self.historical_reports['reports'].append(report)
#             with open(f'{REPORT_DIR}/{self.stock_symbol}_reports.json', 'w') as f:
#                 json.dump(self.historical_reports, f, ensure_ascii=False, indent=2)
            
#             return report
#         except Exception as e:
#             print(f"فشل توليد التقرير لـ {self.stock_symbol}: {str(e)}")
#             return {}

#     def predict_extremes(self, days=30):
#         predictions = []
#         for i in range(days):
#             date = self.today + timedelta(days=i+1)
#             try:
#                 positions = self.calculate_planetary_positions(date)
#                 aspects = self.detect_aspects(positions)
#                 features = self.prepare_features(positions, aspects)
#                 proba = self.model.predict_proba([features])[0]
                
#                 prediction = {
#                     'date': date.strftime("%Y-%m-%d"),
#                     'type': 'peak' if proba[1] > 0.7 else 'trough' if proba[0] > 0.6 else 'neutral',
#                     'confidence': min(round(max(proba)*100, 2), 95.0),  # تحديد سقف الثقة
#                     'critical_angles': self.get_critical_angles(positions),
#                     'aspects': self.get_aspect_details(positions)
#                 }
#                 predictions.append(prediction)
#             except:
#                 continue
#         return predictions

#     def get_critical_angles(self, positions):
#         critical = []
#         thresholds = {
#             'sun': 15, 'moon': 10, 'mercury': 12,
#             'venus': 15, 'mars': 20, 'jupiter': 30
#         }
#         for planet, angle in positions.items():
#             if planet in thresholds:
#                 remainder = angle % thresholds[planet]
#                 if remainder < 2 or (thresholds[planet] - remainder) < 2:
#                     critical.append(f"{planet}_{angle:.1f}")
#         return critical

#     def get_aspect_details(self, positions):
#         aspects = []
#         planets = list(positions.keys())
#         for i in range(len(planets)):
#             for j in range(i+1, len(planets)):
#                 angle = abs(positions[planets[i]] - positions[planets[j]]) % 360
#                 angle = min(angle, 360 - angle)
#                 if angle <= 8:
#                     aspects.append(f"{planets[i]}_conjunction_{planets[j]}")
#                 elif 55 <= angle <= 65:
#                     aspects.append(f"{planets[i]}_sextile_{planets[j]}")
#                 elif 85 <= angle <= 95:
#                     aspects.append(f"{planets[i]}_square_{planets[j]}")
#                 elif 115 <= angle <= 125:
#                     aspects.append(f"{planets[i]}_trine_{planets[j]}")
#                 elif 175 <= angle <= 185:
#                     aspects.append(f"{planets[i]}_opposition_{planets[j]}")
#         return aspects

#     def get_price_stats(self):
#         try:
#             if self.historical_data.empty:
#                 return {'all_time_high': 'N/A', 'all_time_low': 'N/A', 'current_price': 'N/A'}
            
#             return {
#                 'all_time_high': round(self.historical_data['Close'].max().item(), 2),
#                 'all_time_low': round(self.historical_data['Close'].min().item(), 2),
#                 'current_price': round(self.historical_data['Close'].iloc[-1].item(), 2)
#             }
#         except Exception as e:
#             print(f"Error in price stats: {str(e)}")
#             return {'all_time_high': 'N/A', 'all_time_low': 'N/A', 'current_price': 'N/A'}

# # ------------------- التشغيل الرئيسي -------------------
# def generate_daily_reports():
#     stocks = ['2220.SR', '1111.SR']
    
#     for symbol in stocks:
#         try:
#             print(f"\nجارٍ معالجة {symbol}...")
#             reporter = AdvancedStockReporter(symbol)
#             reporter.daily_update()
#             report = reporter.generate_full_report()
            
#             if report:
#                 print(f"\n{'='*50}")
#                 print(f"تقرير {symbol}")
#                 print(f"تاريخ الإدراج: {report.get('listing_date', 'غير معروف')}")
#                 print(f"السعر الحالي: {report.get('price_history_stats', {}).get('current_price', 'N/A')}")
#                 print(f"الدقة: {report.get('model_performance', {}).get('accuracy', 0.0)}%")
                
#         except ValueError as ve:
#             print(f"خطأ في البيانات: {str(ve)}")
#         except KeyError as ke:
#             print(f"مفتاح مفقود: {str(ke)}")
#         except Exception as e:
#             print(f"فشل غير متوقع: {str(e)}")

# schedule.every().day.at("16:00", SAUDI_TZ).do(generate_daily_reports)

# if __name__ == "__main__":
#     print("بدء تشغيل النظام...")
#     generate_daily_reports()
#     # while True:
#     #     schedule.run_pending()
#     #     time.sleep(60)