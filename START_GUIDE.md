# 💎 KOL Tracker ML System - Guía de Inicio Rápido

## 🚀 Iniciar el Sistema Completo

### Un solo comando para TODO:

```bash
python start_all.py
```

Esto iniciará automáticamente:
- ✅ **Tracker** - Escanea trades cada 5 minutos
- ✅ **ML Trainer** - Reentrena modelos cada 1 hora
- ✅ **Token Discovery** - Descubre nuevos traders cada 1 hora
- ✅ **Token Updater** - Actualiza metadata cada 35 minutos
- ✅ **Dashboard** - Interfaz web en http://localhost:8502

## 🛂 Detener el Sistema

```bash
python stop_all.py
```

Esto detendrá TODOS los procesos de forma segura.

## 📊 Dashboard

Una vez iniciado el sistema, accede a:

**http://localhost:8502**

### Tabs del Dashboard:
1. 🔥 **Hot KOLs** - KOLs más activos (últimas 24h)
2. 💎 **Diamond Hands** - Leaderboard con scoring
3. 🕵️ **Discovered** - Traders descubiertos automáticamente
4. 📈 **Gráficos** - Visualizaciones y análisis
5. 🔄 **Recent Trades** - Últimos 20 trades
6. 🔍 **KOL Details** - Estadísticas individuales
7. 🪙 **Tokens** - Tokens trackeados con analytics
8. 📊 **System Overview** - Estado completo del sistema

## ⏰ Intervalos de Actualización

| Proceso | Intervalo |
|---------|-----------|
| 🔍 Tracker | 5 minutos |
| 🧠 ML Trainer | 1 hora |
| 🕵️ Token Discovery | 1 hora |
| 🪙 Token Updater | 35 minutos |

## 📝 Logs

Todos los procesos guardan logs en archivos separados:

- `tracker.log` - Actividad del tracker
- `trainer.log` - Entrenamiento de modelos ML
- `discovery.log` - Descubrimiento de nuevos traders
- `token_updater.log` - Actualización de metadata de tokens
- `dashboard.log` - Logs de Streamlit

## 🔧 Verificar Estado

Para verificar que los procesos están corriendo:

**Windows:**
```bash
tasklist | findstr python
tasklist | findstr streamlit
```

**Linux/Mac:**
```bash
ps aux | grep python
ps aux | grep streamlit
```

## 📌 Archivos Principales

### Scripts de Control:
- `start_all.py` - Inicia TODO con un comando
- `stop_all.py` - Detiene TODO con un comando

### Scripts de Procesos:
- `run_tracker_continuous.py` - Tracker continuo
- `run_continuous_trainer.py` - ML Trainer continuo
- `run_token_discovery_continuous.py` - Discovery continuo
- `run_token_updater_continuous.py` - Token Updater continuo
- `dashboard_unified.py` - Dashboard unificado

## ⚠️ Importante

- **NO cierres la terminal** donde ejecutas `start_all.py` si quieres que el sistema siga corriendo
- Para ejecutar en background, usa:
  ```bash
  python start_all.py &
  ```
- Presiona **Ctrl+C** en la terminal de `start_all.py` para detener TODO

## 🎯 Flujo de Trabajo Típico

1. **Iniciar el sistema:**
   ```bash
   python start_all.py
   ```

2. **Abrir el dashboard:**
   - Navega a http://localhost:8502
   - Explora las diferentes tabs
   - Monitorea Hot KOLs y Diamonds Hands

3. **Dejar corriendo:**
   - El sistema trabaja automáticamente
   - Todos los procesos corren en background
   - Los logs se actualizan en tiempo real

4. **Cuando termines:**
   ```bash
   python stop_all.py
   ```

## 🆘 Troubleshooting

### Dashboard no carga:
- Verifica que no haya otra instancia de Streamlit corriendo
- Ejecuta `python stop_all.py` y luego `python start_all.py` nuevamente

### Procesos no inician:
- Verifica los archivos de log para ver errores
- Asegúrate de estar en el directorio correcto
- Verifica que todas las dependencias estén instaladas

### Puerto 8502 ocupado:
- Cambia el puerto en `start_all.py` agregando `--server.port 8503`

---

**¡El sistema está listo para usar! 🚀**
