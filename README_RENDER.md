# 🚀 KOL Tracker ML - Deployment en Render

Guía completa de deployment en Render.com

---

## 📋 Resumen

**Sistema:** KOL Tracker ML + Bubblemaps Integration
**Plataforma:** Render.com
**Costo estimado:** ~$25-35/mes
**Tiempo de deployment:** 15-20 minutos

---

## 🎯 Arquitectura en Render

```
┌─────────────────────────────────────────────┐
│  Render Dashboard                            │
├─────────────────────────────────────────────┤
│                                              │
│  🌐 Web Service (Dashboard)                 │
│  └─ streamlit dashboard                     │
│                                              │
│  ⚙️ Workers (Procesos Background)           │
│  ├─ Tracker (cada 5 min)                    │
│  ├─ ML Trainer (cada 1 hora)                │
│  ├─ Token Discovery (cada 1 hora)           │
│  └─ Token Updater (cada 35 min)             │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 📦 Pre-Deployment Checklist

### ✅ Antes de empezar, asegúrate de tener:

- [x] Cuenta en GitHub (gratuito)
- [x] Todo el código organizado en carpetas
- [x] `requirements.txt` actualizado
- [x] `render.yaml` configurado
- [x] `.gitignore` creado
- [x] API Keys listas:
  - Helius RPC URL
  - Bubblemaps API Key (opcional)

---

## 🚀 Paso a Paso: Deployment

### **PASO 1: Crear Repositorio en GitHub**

1. **Ve a GitHub:** https://github.com
2. **Crea un nuevo repo:**
   - Name: `kol-tracker-ml`
   - Description: `KOL Tracking System with ML and Bubblemaps`
   - Private: ✅ (recomendado)
3. **NO marques "Initialize with README"** (ya tienes código)

### **PASO 2: Subir código a GitHub**

Desde tu terminal local:

```bash
# Navegar al proyecto
cd "C:\Users\migue\Desktop\claude creaciones\kol_tracker_ml"

# Inicializar git
git init

# Añadir todos los archivos
git add .

# Hacer commit inicial
git commit -m "Initial commit - KOL Tracker ML with Bubblemaps"

# Añadir remote (reemplaza TU_USUARIO)
git remote add origin https://github.com/TU_USUARIO/kol-tracker-ml.git

# Subir a GitHub
git branch -M main
git push -u origin main
```

**Si te pide usuario/password:**
- Usuario: Tu GitHub username
- Password: Personal Access Token (crear en GitHub Settings → Developer Settings → Personal Access Tokens)

### **PASO 3: Crear Cuenta en Render**

1. **Ve a:** https://render.com
2. **Sign up:** "Sign up with GitHub"
3. **Autoriza:** GitHub access
4. **Verifica email**

### **PASO 4: Crear Nuevo Servidor en Render**

Render leerá automáticamente tu `render.yaml` y creará todos los servicios.

**Opción A: Blueprint Automatic (Recomendado)**

1. En el dashboard de Render, clic "New +"
2. Selecciona "Blueprint"
3. Connect to GitHub → Autoriza
4. Selecciona el repo `kol-tracker-ml`
5. Render detectará `render.yaml` automáticamente
6. Clic "Apply Blueprint"

**Opción B: Manual (si falla el automático)**

Crea cada servicio manualmente:

#### **4.1 Dashboard (Web Service)**
- Type: Web Service
- Name: kol-tracker-dashboard
- Environment: Python 3
- Region: Oregon (o el más cercano)
- Branch: main
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run dashboard/dashboard_unified.py --server.port=$PORT --server.address=0.0.0.0`
- Plan: Starter ($7/mes)

#### **4.2 Tracker Worker**
- Type: Worker
- Name: kol-tracker
- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python processes/run_tracker_continuous.py`
- Plan: Starter ($7/mes)

#### **4.3 ML Trainer Worker**
- Type: Worker
- Name: ml-trainer
- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python processes/run_continuous_trainer.py`
- Plan: Starter ($7/mes)

#### **4.4 Token Discovery Worker**
- Type: Worker
- Name: token-discovery
- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python discovery/run_token_discovery_continuous.py`
- Plan: Starter ($7/mes)

#### **4.5 Token Updater Worker**
- Type: Worker
- Name: token-updater
- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python processes/run_token_updater_both_continuous.py`
- Plan: Starter ($7/mes)

### **PASO 5: Configurar Variables de Entorno**

En cada servicio, añade las siguientes Environment Variables:

**Para todos los servicios:**
```
PYTHON_VERSION=3.11.0
```

**Para workers que usan RPC:**
```
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=TU_API_KEY
```

**Para token updater:**
```
BUBBLEMAPS_API_KEY=TU_API_KEY
```

### **PASO 6: Deployment!**

Render comenzará a:
1. Build el proyecto
2. Instalar dependencias
3. Iniciar los servicios
4. Mostrar logs en tiempo real

**Tiempo estimado:** 5-10 minutos por servicio

### **PASO 7: Verificar Deployment**

Cuando termine, verás:
- ✅ "Live" en verde
- URL del Dashboard
- Logs en tiempo real

**URLs típicas:**
- Dashboard: `https://kol-tracker-dashboard.onrender.com`
- Logs: Clic en el servicio → "Logs"

---

## 📊 Monitoreo

### **Ver Logs**
1. Dashboard de Render
2. Clic en el servicio
3. Tab "Logs"
4. Ver logs en tiempo real

### **Métricas**
- CPU usage
- Memory usage
- Response times
- Uptime

### **Reiniciar servicios**
Si algo falla:
1. Clic en el servicio
2. "Manual Deploy"
3. "Deploy latest commit"

---

## 💰 Costos

| Servicio | Tipo | Plan | Costo/mes |
|----------|------|------|-----------|
| Dashboard | Web Service | Starter | $7 |
| Tracker | Worker | Starter | $7 |
| ML Trainer | Worker | Starter | $7 |
| Token Discovery | Worker | Starter | $7 |
| Token Updater | Worker | Starter | $7 |
| **TOTAL** | | | **$35/mes** |

**Consejo:** Empieza con 2-3 workers esenciales (Tracker + Updater + Dashboard) = $21/mes

---

## 🔄 Actualizar el Sistema

### **Hacer cambios:**

1. **Cambios en código:**
```bash
git add .
git commit -m "Descripción del cambio"
git push
```

2. **Render auto-deploy:**
   - Detecta el push
   - Reinicia automáticamente
   - No requiere acción manual

### **Forzar redeploy:**
- Dashboard Render → Servicio → "Manual Deploy"

---

## 🐛 Troubleshooting

### **Problema: Service crashes on startup**

**Solución:**
1. Ver logs completos
2. Buscar errores de import
3. Verificar que todos los archivos estén en el repo
4. Chequear `requirements.txt`

### **Problema: Out of memory**

**Solución:**
- Upgrade plan (Starter → Standard)
- Optimizar código
- Reducir batch sizes

### **Problema: Workers sleeping**

**Nota:** Render workers no se duermen (solo free tier de web services)

### **Problema: Database locked**

**Solución:**
- SQLite tiene limitaciones en entornos cloud
- Considerar migrar a PostgreSQL (Render tiene PostgreSQL gratis)

---

## 🔒 Seguridad

### **API Keys en Render**

1. **Nunca** hardcodear API keys en código
2. Usar siempre Environment Variables
3. Marcar variables como "sensitive" en Render

### **Ejemplo:**
```python
# ❌ MAL
API_KEY = "abc123"

# ✅ BIEN
import os
API_KEY = os.getenv("BUBBLEMAPS_API_KEY")
```

---

## 📈 Escalar

### **Más recursos:**
1. Dashboard Render → Servicio
2. Settings → Deploy
3. Cambiar plan (Starter → Standard → Pro)

### **Más instancias:**
1. Settings → Scale
2. Aumentar número de instancias

---

## ✅ Post-Deployment Checklist

- [x] Dashboard accesible en URL pública
- [x] Todos los workers corriendo
- [x] Logs sin errores críticos
- [x] Base de datos creciendo
- [x] Trades siendo recolectados
- [x] ML models entrenando periódicamente

---

## 🎓 Recursos

- **Render Docs:** https://render.com/docs
- **Render YAML:** https://render.com/docs/yaml-spec
- **Python on Render:** https://render.com/docs/deploy-python-example

---

## 🆘 Soporte

Si algo falla:
1. Ver logs en Render
2. Check este README
3. Render community: https://community.render.com
4. Email: support@render.com

---

**Última actualización:** 2026-01-13
**Estado:** ✅ Production Ready
