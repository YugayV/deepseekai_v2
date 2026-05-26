"""
МОДУЛЬ МАШИННОГО ОБУЧЕНИЯ
Классификация направления и регрессия цены EUR/USD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            mean_absolute_error, mean_squared_error, r2_score)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

class MLPipeline:
    """Пайплайн машинного обучения"""
    
    def __init__(self, features_df):
        self.features_df = features_df.copy()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, task='classification'):
        """
        Подготовка данных для обучения
        
        Args:
            task: 'classification' или 'regression'
        """
        print(f"\n🔧 Подготовка данных для задачи: {task}")
        
        # Целевая переменная
        if task == 'classification':
            target_col = 'Target_Direction'
        else:
            target_col = 'Target_Return'
        
        # Удаляем целевые переменные из признаков
        feature_cols = [col for col in self.features_df.columns 
                       if not col.startswith('Target')]
        
        X = self.features_df[feature_cols].copy()
        y = self.features_df[target_col].copy()
        
        # Удаляем строки с NaN в целевой переменной
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"  📊 Признаков: {X.shape[1]}")
        print(f"  📅 Примеров: {X.shape[0]}")
        
        if task == 'classification':
            print(f"  🎯 Распределение классов:")
            print(f"     Down (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
            print(f"     Up   (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        
        # Time-series split: train 80%, test 20%
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"  🚂 Train: {len(X_train)} примеров")
        print(f"  🧪 Test:  {len(X_test)} примеров")
        
        # Нормализация
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, 
                X_train.index, X_test.index, feature_cols)
    
    def train_classification_models(self):
        """Обучение моделей классификации"""
        print("\n" + "="*70)
        print("🎯 КЛАССИФИКАЦИЯ НАПРАВЛЕНИЯ (UP/DOWN)")
        print("="*70)
        
        # Подготовка данных
        (X_train, X_test, y_train, y_test, 
         train_idx, test_idx, feature_cols) = self.prepare_data('classification')
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                   random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, 
                                        learning_rate=0.1, random_state=42)
        }
        
        classification_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"🤖 Модель: {model_name}")
            print(f"{'='*70}")
            
            # Обучение
            print("  ⏳ Обучение...")
            model.fit(X_train, y_train)
            
            # Предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Метрики
            print("\n  📊 МЕТРИКИ:")
            print("  " + "-"*66)
            print(f"  {'Метрика':<25} {'Train':>15} {'Test':>15}")
            print("  " + "-"*66)
            
            metrics = {}
            for dataset_name, y_true, y_pred in [('Train', y_train, y_train_pred),
                                                 ('Test', y_test, y_test_pred)]:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                metrics[dataset_name] = {
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1
                }
            
            for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                train_val = metrics['Train'][metric_name]
                test_val = metrics['Test'][metric_name]
                print(f"  {metric_name:<25} {train_val:>15.4f} {test_val:>15.4f}")
            
            print("  " + "-"*66)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print("\n  📊 CONFUSION MATRIX (Test):")
            print(f"       Predicted Down  Predicted Up")
            print(f"  Down      {cm[0,0]:<10}  {cm[0,1]:<10}")
            print(f"  Up        {cm[1,0]:<10}  {cm[1,1]:<10}")
            
            # Feature Importance (если доступно)
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                print("\n  ⭐ TOP-10 ВАЖНЫХ ПРИЗНАКОВ:")
                for idx, row in importance.iterrows():
                    print(f"     {row['Feature']:<30} {row['Importance']:.4f}")
            
            # Сохранение результатов
            classification_results[model_name] = {
                'model': model,
                'train_metrics': metrics['Train'],
                'test_metrics': metrics['Test'],
                'y_test': y_test,
                'y_test_pred': y_test_pred,
                'test_index': test_idx,
                'confusion_matrix': cm
            }
            
            if hasattr(model, 'feature_importances_'):
                classification_results[model_name]['feature_importance'] = importance
        
        self.results['classification'] = classification_results
        return classification_results
    
    def train_regression_models(self):
        """Обучение моделей регрессии"""
        print("\n" + "="*70)
        print("📈 РЕГРЕССИЯ ДОХОДНОСТИ (PRICE RETURN)")
        print("="*70)
        
        # Подготовка данных
        (X_train, X_test, y_train, y_test, 
         train_idx, test_idx, feature_cols) = self.prepare_data('regression')
        
        models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                                  random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
        }
        
        regression_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"🤖 Модель: {model_name}")
            print(f"{'='*70}")
            
            # Обучение
            print("  ⏳ Обучение...")
            model.fit(X_train, y_train)
            
            # Предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Метрики
            print("\n  📊 МЕТРИКИ:")
            print("  " + "-"*66)
            print(f"  {'Метрика':<25} {'Train':>15} {'Test':>15}")
            print("  " + "-"*66)
            
            metrics = {}
            for dataset_name, y_true, y_pred in [('Train', y_train, y_train_pred),
                                                 ('Test', y_test, y_test_pred)]:
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                metrics[dataset_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2
                }
            
            for metric_name in ['MAE', 'RMSE', 'R²']:
                train_val = metrics['Train'][metric_name]
                test_val = metrics['Test'][metric_name]
                print(f"  {metric_name:<25} {train_val:>15.4f} {test_val:>15.4f}")
            
            print("  " + "-"*66)
            
            # Направленная точность (правильное предсказание знака)
            direction_train = ((y_train_pred > 0) == (y_train > 0)).mean()
            direction_test = ((y_test_pred > 0) == (y_test > 0)).mean()
            print(f"\n  🎯 Точность направления:")
            print(f"     Train: {direction_train:.2%}")
            print(f"     Test:  {direction_test:.2%}")
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                print("\n  ⭐ TOP-10 ВАЖНЫХ ПРИЗНАКОВ:")
                for idx, row in importance.iterrows():
                    print(f"     {row['Feature']:<30} {row['Importance']:.4f}")
            
            # Сохранение результатов
            regression_results[model_name] = {
                'model': model,
                'train_metrics': metrics['Train'],
                'test_metrics': metrics['Test'],
                'y_test': y_test,
                'y_test_pred': y_test_pred,
                'test_index': test_idx,
                'direction_accuracy': direction_test
            }
            
            if hasattr(model, 'feature_importances_'):
                regression_results[model_name]['feature_importance'] = importance
        
        self.results['regression'] = regression_results
        return regression_results
    
    def save_models(self, output_dir='/home/ubuntu/eurusd_analysis/models'):
        """Сохранение моделей"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for task, models_dict in self.results.items():
            for model_name, results in models_dict.items():
                model_filename = f"{task}_{model_name.replace(' ', '_')}.pkl"
                joblib.dump(results['model'], f"{output_dir}/{model_filename}")
                print(f"💾 Сохранена модель: {model_filename}")
        
        # Сохраняем scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        print(f"💾 Сохранен scaler: scaler.pkl")
    
    def save_results_csv(self, output_dir='/home/ubuntu/eurusd_analysis/results'):
        """Сохранение результатов в CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Сводная таблица метрик
        metrics_data = []
        
        if 'classification' in self.results:
            for model_name, results in self.results['classification'].items():
                metrics_data.append({
                    'Task': 'Classification',
                    'Model': model_name,
                    'Accuracy_Train': results['train_metrics']['Accuracy'],
                    'Accuracy_Test': results['test_metrics']['Accuracy'],
                    'Precision_Test': results['test_metrics']['Precision'],
                    'Recall_Test': results['test_metrics']['Recall'],
                    'F1_Test': results['test_metrics']['F1-Score']
                })
        
        if 'regression' in self.results:
            for model_name, results in self.results['regression'].items():
                metrics_data.append({
                    'Task': 'Regression',
                    'Model': model_name,
                    'MAE_Train': results['train_metrics']['MAE'],
                    'MAE_Test': results['test_metrics']['MAE'],
                    'RMSE_Test': results['test_metrics']['RMSE'],
                    'R2_Test': results['test_metrics']['R²'],
                    'Direction_Accuracy': results['direction_accuracy']
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{output_dir}/ml_metrics_summary.csv", index=False)
        print(f"\n💾 Сохранена сводная таблица метрик: ml_metrics_summary.csv")


def main():
    """Главная функция"""
    print("\n" + "="*70)
    print("🤖 МАШИННОЕ ОБУЧЕНИЕ - EUR/USD PREDICTION")
    print("="*70)
    
    # Загрузка признаков
    features_df = pd.read_csv('/home/ubuntu/eurusd_analysis/data/features.csv',
                             index_col=0, parse_dates=True)
    
    print(f"\n📊 Загружено: {len(features_df)} примеров, {len(features_df.columns)} признаков")
    
    # Создание пайплайна
    pipeline = MLPipeline(features_df)
    
    # Обучение моделей классификации
    classification_results = pipeline.train_classification_models()
    
    # Обучение моделей регрессии
    regression_results = pipeline.train_regression_models()
    
    # Сохранение моделей и результатов
    pipeline.save_models()
    pipeline.save_results_csv()
    
    print("\n" + "="*70)
    print("✅ МАШИННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*70)
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
