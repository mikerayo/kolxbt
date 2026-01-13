# 🧹 LIMPIEZA DE PROYECTO - ANÁLISIS DE ARCHIVOS

## 📊 Resumen
**Total archivos Python:** 62
**Archivos ACTIVOS:** 30
**Archivos OBSOLETOS:** 15
**Archivos de TEST:** 10
**Archivos DUPLICADOS:** 7

---

## ✅ ARCHIVOS ACTIVOS (NO BORRAR)

### Core System
```
database.py                    # Modelos de base de datos
wallet_tracker.py              # Lógica principal de tracking
transaction_parser.py          # Parseo de transacciones
feature_engineering.py         # Cálculo de features
ml_models.py                   # Modelos de ML
analyzer.py                    # Análisis y reportes
config.py                      # Configuración
utils.py                       # Utilidades
```

### APIs (Integraciones externas)
```
bubblemaps_api.py              # Bubblemaps API ✨ NUEVO
dexscreener_api.py             # DexScreener API
pumpfun_parser.py              # Pump.fun parser
```

### Procesos Continuos (ACTIVOS)
```
run_tracker_continuous.py               # Tracker cada 5 min
run_continuous_trainer.py               # ML Trainer cada 1 hora
run_token_discovery_continuous.py       # Discovery cada 1 hora
run_token_updater_both_continuous.py    # Token updater cada 35 min ✨ NUEVO
run_analyzer_continuous.py              # Analyzer continuo
run_hot_kols.py                         # Hot KOLs updater
run_summary_scheduler.py                # Scheduler de resúmenes
```

### Dashboard
```
dashboard_unified.py          # Dashboard principal ✅ ÚNICO ACTIVO
```

### Master Launcher
```
start_all.py                  # Arranca todo
stop_all.py                   # Detiene todo
```

### Data Updaters
```
update_tokens_both.py         # Actualiza tokens (DexScreener + Bubblemaps) ✨ NUEVO
```

### Discovery System
```
token_centric_discovery.py    # Discovery basado en tokens
run_discovery_continuous.py   # Runner de discovery
```

### Análisis
```
hot_kols_scorer.py            # Scoring de Hot KOLs
wallet_analyzer.py            # Análisis de wallets
```

---

## ❌ ARCHIVOS OBSOLETOS (SE PUEDEN BORRAR)

### Dashboards Antiguos
```
❌ dashboard.py              # Reemplazado por dashboard_unified.py
❌ dashboard_v2.py           # Reemplazado por dashboard_unified.py
```

### Token Updaters Antiguos
```
❌ update_tokens.py          # Reemplazado por update_tokens_both.py
❌ run_token_updater_continuous.py  # Reemplazado por run_token_updater_both_continuous.py
```

### Discovery Antiguo
```
❌ token_buyer_discovery.py  # Reemplazado por token_centric_discovery.py
```

### Trackers Antiguos
```
❌ run_tracker.py            # Reemplazado por run_tracker_continuous.py
❌ run_tracker_incremental.py # Función integrada en run_tracker_continuous.py
```

### Parser Antiguo
```
❌ enhanced_parser.py        # Lógica integrada en transaction_parser.py
```

### Scripts One-Time (Ya no se usan)
```
❌ analyze_tx.py             # Script manual de análisis
```

---

## 🧪 ARCHIVOS DE TEST (Mover a carpeta /tests)

```
test_bubblemaps_direct.py         # Test Bubblemaps
test_bubblemaps_integration.py    # Test integración Bubblemaps
test_discovery.py                 # Test discovery
test_full_integration.py          # Test integración completa
test_pumpfun.py                   # Test pump.fun parser
test_small.py                     # Test pequeño
test_tracker.py                   # Test tracker
```

**Acción:** Crear carpeta `tests/` y mover estos archivos allí

---

## 🔍 ARCHIVOS DE DEBUG (Mover a carpeta /debug)

```
debug_pumpfun.py            # Debug de pump.fun
```

**Acción:** Crear carpeta `debug/` y mover

---

## 📁 ESTRUCTURA PROPUESTA

### Estructura Limpia:
```
kol_tracker_ml/
├── CORE/
│   ├── database.py
│   ├── wallet_tracker.py
│   ├── transaction_parser.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   ├── analyzer.py
│   ├── config.py
│   └── utils.py
│
├── apis/
│   ├── bubblemaps_api.py
│   ├── dexscreener_api.py
│   └── pumpfun_parser.py
│
├── processes/
│   ├── run_tracker_continuous.py
│   ├── run_continuous_trainer.py
│   ├── run_token_discovery_continuous.py
│   ├── run_token_updater_both_continuous.py
│   ├── run_analyzer_continuous.py
│   ├── run_hot_kols.py
│   └── run_summary_scheduler.py
│
├── discovery/
│   ├── token_centric_discovery.py
│   ├── run_discovery_continuous.py
│   └── hot_kols_scorer.py
│
├── dashboard/
│   └── dashboard_unified.py
│
├── launchers/
│   ├── start_all.py
│   └── stop_all.py
│
├── updaters/
│   └── update_tokens_both.py
│
├── tests/
│   ├── test_bubblemaps_direct.py
│   ├── test_bubblemaps_integration.py
│   ├── test_discovery.py
│   ├── test_full_integration.py
│   ├── test_pumpfun.py
│   ├── test_small.py
│   └── test_tracker.py
│
├── debug/
│   └── debug_pumpfun.py
│
└── data/
    └── kol_tracker.db
```

---

## 🗑️ ARCHIVOS A BORRAR (15 archivos)

```bash
# Dashboards antiguos
dashboard.py
dashboard_v2.py

# Token updaters antiguos
update_tokens.py
run_token_updater_continuous.py

# Discovery antiguo
token_buyer_discovery.py

# Trackers antiguos
run_tracker.py
run_tracker_incremental.py

# Parser antiguo
enhanced_parser.py

# Scripts one-time
analyze_tx.py

# Discovery simple (reemplazado)
simple_discovery.py
```

---

## ⚡ ACCIÓN INMEDIATA

¿Quieres que yo:

**Opción 1: Reorganizar completo** 🎁
- Crear estructura de carpetas (/core, /apis, /processes, etc.)
- Mover archivos a sus carpetas correspondientes
- Actualizar imports en todos los archivos
- Borrar obsoletos

**Opción 2: Solo borrar obsoletos** 🧹
- Borrar solo los 15 archivos obsoletos
- Mantener estructura actual (todo en raíz)
- Mover tests a /tests

**Opción 3: Crear script de limpieza** 📜
- Crear un script que tú puedes ejecutar
- Te deja decidir qué borrar

¿Cuál prefieres? 🚀
