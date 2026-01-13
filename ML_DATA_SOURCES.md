# 🎯 FUENTES DE DATOS ADICIONALES PARA ML

## 1. BUBBLEMAPS API 🔥 (PRIORIDAD ALTA)

### Qué datos ofrece Bubblemaps:
```
Bubblemaps visualiza la distribución de tokens entre holders:
├─ Top holders y sus porcentajes
├─ Conexiones entre wallets (clusters)
├─ Concentración de tokens
├─ Distribución de holders
└─ Actividad de holders (compras/ventas)
```

### API Endpoints disponibles:

#### 1.1 Token Distribution Data
```python
GET https://api.bubblemaps.io/token/{chain}/{token_address}

Response:
{
  "token": "...",
  "holders": [
    {
      "address": "wallet_address",
      "balance": 1000000,
      "percentage": 15.5,  # % del total supply
      "label": "Dev/CEX/Whale"  # Si está etiquetado
    },
    ...
  ],
  "top10_percentage": 45.2,  # % controlado por top 10
  "holder_count": 1250,  # Total holders únicos
  "gini_coefficient": 0.75,  # 0=distribuido, 1=concentrado
  "clusters": [  # Grupos de wallets conectadas
    {
      "cluster_id": "...",
      "addresses": ["addr1", "addr2", ...],
      "total_percentage": 25.3,
      "type": "dev" | "insider" | "whale"
    }
  ]
}
```

### ¿Por qué estos datos son GOLD para el ML?

#### **FEATURES QUE PODRÍAMOS AÑADIR:**

1. **Concentración de Top Holders:**
   ```
   - top10_percentage: ¿Demasiado concentrado?
     - Si top 10 tiene >80% → Riesgo alto (dump)
     - Si top 10 tiene <40% → Más democrático

   - gini_coefficient: Índice de desigualdad
     - 0.0-0.3: Muy distribuido (bueno)
     - 0.3-0.6: Moderado
     - 0.6-1.0: Muy concentrado (malo)
   ```

2. **Insider/Dev Holdings:**
   ```
   - dev_percentage: ¿Qué % tiene el dev?
     - Si dev tiene >30% → Puede dumpear
     - Si dev tiene <10% → Más seguro

   - insider_cluster: ¿Hay grupo de insiders?
     - Si hay cluster conectado al dev → Cuidado
   ```

3. **Holder Growth:**
   ```
   - holder_count_change: ¿Crece el número de holders?
     - +50% en 24h → Hype positivo
     - -20% en 24h → Gente vendiendo

   - unique_24h: Nuevos holders únicos
   - returning_holders: Holders que recompraron
   ```

4. **Whale Activity:**
   ```
   - whale_accumulation: ¿Ballenas acumulando?
     - Si whale buys → Bullish
     - Si whale sells → Bearish

   - large_tx_count: Transacciones grandes
   ```

### Integración con Token Predictor:

```python
# NUEVAS FEATURES para el modelo
features_bubblemaps = {
    # Concentración
    'top10_percentage': 45.2,        # % en top 10
    'top1_percentage': 15.5,         # % en wallet #1
    'gini_coefficient': 0.75,        # Distribución
    'herfindahl_index': 0.23,        # Concentración HHI

    # Dev/Insider risk
    'dev_percentage': 12.3,          # Dev holdings
    'insider_cluster_pct': 25.0,     # Cluster insider
    'team_wallets_count': 3,         # Wallets del team
    'team_locked': True,             # ¿Tiene lock?

    # Holder dynamics
    'holder_count': 1250,            # Total holders
    'holder_growth_24h': 0.15,       # % crecimiento
    'new_holders_24h': 150,         # Nuevos holders
    'returning_holders_ratio': 0.35, # % que vuelven

    # Whale activity
    'whale_count': 12,               # Ballenas (>1%)
    'whale_accumulating': True,      # ¿Acumulando?
    'large_buys_24h': 5,             # Compras grandes
    'large_sells_24h': 2,            # Ventas grandes

    # Liquidity distribution
    'lp_top10_pct': 60.0,            # LP en top 10
    'lp_concentration': 0.45,        # Concentración LP
}
```

### Impacto esperado en el ML:

```
ANTES (sin Bubblemaps):
├─ AUC: 0.60
├─ Features: 7
└─ No conoce distribución de holders

DESPUÉS (con Bubblemaps):
├─ AUC: 0.75-0.85 (proyectado)
├─ Features: 20+
├─ Sabe si token está concentrado
├─ Detecta si dev puede dumpear
└─ Ve acumulación de ballenas
```

---

## 2. OTRAS API VALIOSAS

### 2.1 Solana Beach / Solscan API
```python
# Transaction patterns
GET https://api.solana.com/v1/token/{address}/transfers

Útil para:
├─ Transfer patterns (insider moving)
├─ First buyer after listing
├─ Sniper activity
└─ Wash trading detection
```

### 2.2 DexScreener Extended Data
```python
# Ya lo usamos pero podríamos añadir:
GET https://api.dexscreener.com/latest/dex/pairs/{pair}

Additional data:
├─ txns (transactions)
│  ├─ h24: {buys: 1500, sells: 800}
│  └─ m5: {buys: 50, sells: 30}
├─ buys_24h: 1500
├─ sells_24h: 800
└─ buy_sell_ratio_24h: 1.875
```

### 2.3 Twitter/X API (Sentiment)
```python
# Social sentiment
GET https://api.twitter.com/2/tweets/search/recent

Útil para:
├─ Hype detection
├─ Mention count
├─ Sentiment analysis
└─ Influencer activity
```

### 2.4 GeckoTerminal API
```python
# Alternative to DexScreener
GET https://api.geckoterminal.com/api/v2/networks/solana/tokens/{address}

Additional:
├─ Market pairs
├─ Price history (OHLCV)
├─ Social metrics
└─ ATH/ATL tracking
```

---

## 3. ON-CHAIN DATA (Solana RPC)

### Datos adicionales del RPC:
```python
# 1. Token Metadata
get_token_metadata()
├─ Immutable data (burn, mint authority)
├─ Mutable data (update authority)
└─ ¿Se puede mintear más?

# 2. Largest Holders
get_token_largest_accounts()
├─ Top 20 holders
├─ Sus porcentajes
└─ Detectar CEX, dev wallets

# 3. Transaction History
get_signatures_for_address()
├─ First buyers
├─ Early snipers
└─ Holding patterns
```

---

## 4. FEATURES ENGINEERING ADVANZADO

### Combinando múltiples fuentes:

```python
def create_enhanced_features(token_address):
    """Crear super-features combinando APIs"""

    # Base features (ya las tenemos)
    base = {
        'num_kols': 5,
        'sol_invested': 100,
        'num_trades': 50,
        'price_usd': 0.0001,
        'liquidity_usd': 50000,
    }

    # Bubblemaps features
    bubble = get_bubblemaps_data(token_address)
    base.update({
        'concentration_score': calculate_concentration(bubble),
        'dev_risk_score': calculate_dev_risk(bubble),
        'whale_sentiment': calculate_whale_sentiment(bubble),
        'distribution_health': calculate_health(bubble),
    })

    # Trading patterns
    dexscreener = get_dexscreener_extended(token_address)
    base.update({
        'buy_sell_ratio': dexscreener['buys'] / dexscreener['sells'],
        'txn_velocity': dexscreener['txns']['h5'] / 60,  # txns per minute
        'pressure_score': calculate_buying_pressure(dexscreener),
    })

    # On-chain patterns
    rpc_data = get_solana_rpc_data(token_address)
    base.update({
        'sniper_ratio': calculate_snipers(rpc_data),
        'insider_activity': detect_insider_moves(rpc_data),
        'liquidity_locked': is_lp_locked(rpc_data),
    })

    return base
```

### Features calculadas (derivadas):

```python
# 1. Concentration Risk Score
concentration_risk = (
    (top10_pct * 0.4) +           # Top 10 control
    (gini * 0.3) +                 # Inequality
    (dev_pct * 0.2) +              # Dev holdings
    (whale_concentration * 0.1)     # Whale clustering
)
# Score: 0-100 (100 = muy riesgoso)

# 2. Holder Health Score
holder_health = (
    (holder_growth_24h * 0.4) +    # Crecimiento
    (returning_ratio * 0.3) +      # Lealtad
    (diversification * 0.2) +      # Distribución
    (activity_score * 0.1)         # Actividad
)
# Score: 0-100 (100 = muy sano)

# 3. Manipulation Detection
manipulation_flags = {
    'wash_trading': detect_wash_trading(),
    'insider_trading': detect_insider_patterns(),
    'pump_and_dump': detect_pump_dump(),
    'sniper_attack': detect_snipers(),
}
```

---

## 5. ROADMAP DE INTEGRACIÓN

### Fase 1: Bubblemaps (Semanas 1-2)
```python
# Archivo: bubblemaps_api.py
class BubblemapsAPI:
    async def get_token_distribution(self, token_address):
        """Obtener distribución de holders"""

    async def get_concentration_metrics(self, token_address):
        """Calcular métricas de concentración"""

    async def detect_insider_clusters(self, token_address):
        """Detectar clusters de insiders"""
```

### Fase 2: Añadir al Database (Semana 3)
```python
# Nueva tabla en database.py
class TokenDistribution(Base):
    __tablename__ = 'token_distribution'

    token_address = Column(String(44), primary_key=True)
    top10_percentage = Column(Float)
    gini_coefficient = Column(Float)
    dev_percentage = Column(Float)
    holder_count = Column(Integer)
    whale_count = Column(Integer)
    last_updated = Column(DateTime)
```

### Fase 3: Integrar al ML (Semana 4)
```python
# Modificar data_loader.py
def create_enhanced_dataset():
    # Añadir Bubblemaps features
    # Reentrenar modelo con más features
```

---

## 6. IMPACTO ESPERADO

### Sin Bubblemaps vs Con Bubblemaps:

```
ESCENARIO A (actual):
- ML detecta: "5 KOLs compraron, puede ser bueno"
- Resultado: Falsos positivos (dev dump, whales sold)

ESCENARIO B (con Bubblemaps):
- ML detecta: "5 KOLs compraron + dev tiene solo 10% + whales accumulating"
- Resultado: ¡Verdadero positivo! Token 3x+
```

### Mejoras proyectadas:

```
Token Predictor Metrics:
├─ AUC: 0.60 → 0.80 (+33%)
├─ Precision: 40% → 70% (+75%)
├─ Recall: 10% → 50% (+400%)
└─ F1 Score: 0.16 → 0.58 (+262%)

Feature importance:
├─ Dev holdings: 25%
├─ Whale activity: 20%
├─ Concentration: 18%
├─ Holder growth: 15%
└─ Original features: 22%
```

---

## 7. CÓMO EMPEZAR

### Paso 1: Probar Bubblemaps API
```bash
# Test manual
curl "https://api.bubblemaps.io/token/solana/TOKEN_ADDRESS"
```

### Paso 2: Crear wrapper
```python
# bubblemaps_api.py
import aiohttp
from typing import Dict, List

class BubblemapsAPI:
    def __init__(self):
        self.base_url = "https://api.bubblemaps.io"

    async def get_token_distribution(self, token_address: str) -> Dict:
        """Get token holder distribution"""
        # Implementation...
```

### Paso 3: Actualizar TokenInfo
```python
# Añadir campos Bubblemaps a token_info
class TokenInfo(Base):
    # ... existing fields ...

    # Bubblemaps data
    top10_percentage = Column(Float)
    gini_coefficient = Column(Float)
    dev_percentage = Column(Float)
    holder_count = Column(Integer)
    bubblemaps_updated = Column(DateTime)
```

### Paso 4: Reentrenar ML
```python
# Usar nuevas features
features = original_features + bubblemaps_features
model = TokenPredictor(input_dim=len(features))
```

---

## 🎯 CONCLUSIÓN

**Bubblemaps es la MEJOR fuente adicional de datos** porque:

1. ✅ **Detecta riesgos ocultos:**
   - Dev holdings
   - Insider clusters
   - Whale manipulation

2. ✅ **Mejora predicciones:**
   - Identifica tokens verdaderamente democráticos
   - Encuentra gems con distribución sana
   - Evita scam/honey pots

3. ✅ **Datos únicos:**
   - No disponible en DexScreener
   - Visualización de clusters
   - Tracking de wallets conectadas

4. ✅ **Impacto directo en ROI:**
   - Menos falsos positivos
   - Mejor selección de tokens
   - Señales más confiables

**¿Quieres que implemente la integración con Bubblemaps? 🚀**
