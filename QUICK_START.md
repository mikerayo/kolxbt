# Quick Start - Comandos Rápidos

## 🚀 Iniciar Sistema (4 terminales)

```bash
# Terminal 1: Tracker
python kol_tracker.py

# Terminal 2: Analyzer
python analyzer.py

# Terminal 3: Dashboard
streamlit run dashboard.py

# Terminal 4: Continuous Training (opcional)
python run_continuous_trainer.py --interval 6
```

## 🏋️ Entrenar Modelos

```bash
# Entrenar Token Predictor (una vez)
python run_continuous_trainer.py --model token --once --epochs 30 --batch-size 8

# Entrenar KOL Predictor
python run_continuous_trainer.py --model kol --once --epochs 30 --batch-size 8

# Entrenar ambos
python run_continuous_trainer.py --once --epochs 30
```

## 🔮 Hacer Predicciones (Python)

```python
from api.batch_predictor import BatchPredictor
from datetime import datetime

predictor = BatchPredictor()

# Predicción de token
prob = predictor.predict_token_3x_probability(
    kol_id=1,
    amount_sol=10.0,
    entry_time=datetime.now()
)
print(f"Probabilidad 3x+: {prob:.2%}")

# Predicción de KOL
perf_7d, perf_30d = predictor.predict_kol_future_performance(kol_id=1)
print(f"7d: {perf_7d:.2%}, 30d: {perf_30d:.2%}")
```

## 📊 Estado Actual

| Componente | Estado | Métricas |
|-----------|--------|----------|
| Token Predictor | ✅ Entrenado | AUC: 0.7266 |
| KOL Predictor | ⚠️ Esperando datos | Necesita más historia |
| Dashboard | ✅ Funcionando | http://localhost:8501 |

## 📁 Archivos Clave

- `INSTRUCTIONS.md` - Guía completa
- `models/token_predictor_best.pth` - Modelo entrenado
- `database/kol_tracker.db` - Base de datos
- `config.py` - Configuración

## ⚠️ Problema Conocido

Datos desbalanceados: Solo 4 de 1089 posiciones son 3x+ (0.4%)
Solución: Esperar más datos + usar class weights (ver INSTRUCTIONS.md)
