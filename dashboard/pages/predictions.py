"""
ML Predictions Page

Machine Learning predictions with:
- Token 3x+ probability
- KOL future performance predictions
- Model metrics and feature importance
- Anomaly detection
- What-if analysis simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager


def check_model_availability():
    """Check if ML models are trained and available"""
    try:
        from pathlib import Path

        model_path = Path('models/token_predictor_best.pth')
        metadata_path = Path('models/token_predictor_best_metadata.json')

        token_model_exists = model_path.exists() and metadata_path.exists()

        return {
            'token_predictor': token_model_exists,
            'kol_predictor': False,  # Not enough data yet
            'models_available': token_model_exists
        }
    except Exception:
        return {
            'token_predictor': False,
            'kol_predictor': False,
            'models_available': False
        }


def render_token_probability_table():
    """Render table with recent trades and 3x+ probabilities"""
    st.markdown("### 🎯 Token 3x+ Probability")

    # Check model availability
    models = check_model_availability()

    if not models['token_predictor']:
        st.warning("⚠️ Token Predictor model not trained yet")
        st.info("💡 Train the model first:")
        st.code("python run_continuous_trainer.py --model token --once --epochs 30", language="bash")
        return

    data_manager = get_data_manager()

    # Get recent trades
    trades = data_manager.get_recent_trades(limit=50)

    if not trades:
        st.warning("⏳ No recent trades available")
        return

    # Load predictor
    try:
        from api.batch_predictor import BatchPredictor

        predictor = BatchPredictor()

        # Calculate probabilities for recent trades
        predictions = []

        for trade in trades:
            try:
                # Get 3x+ probability
                prob = predictor.predict_token_3x_probability(
                    kol_id=trade['kol_id'],
                    amount_sol=trade['amount_sol'],
                    entry_time=trade['timestamp']
                )

                predictions.append({
                    'KOL': trade['kol_name'],
                    'Token': trade['token_address'][:8] + '...',
                    'Amount SOL': f"{trade['amount_sol']:.2f}",
                    'Time': trade['timestamp'].strftime('%H:%M'),
                    '3x+ Probability': prob,
                    'Confidence': 'High' if prob > 0.7 or prob < 0.3 else 'Medium'
                })
            except Exception as e:
                # Skip trades with errors
                continue

        if not predictions:
            st.warning("⏳ No predictions available")
            return

        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)

        # Display table with color coding
        st.markdown("#### Recent Trades with 3x+ Probability")

        # Format for display
        display_df = pred_df.copy()
        display_df['3x+ Probability'] = (display_df['3x+ Probability'] * 100).round(1).astype(str) + '%'

        # Add color indicators
        def color_probability(val):
            prob = float(val.strip('%')) / 100
            if prob >= 0.7:
                return '🔴 **HIGH**'
            elif prob >= 0.4:
                return '🟡 **MEDIUM**'
            else:
                return '🟢 **LOW**'

        display_df['Signal'] = display_df['3x+ Probability'].apply(color_probability)

        # Reorder columns
        display_df = display_df[['KOL', 'Token', 'Amount SOL', 'Time', '3x+ Probability', 'Signal']]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Show stats
        col1, col2, col3 = st.columns(3)

        probs = [p['3x+ Probability'] for p in predictions]
        avg_prob = np.mean(probs)
        high_prob_count = sum(1 for p in probs if p >= 0.7)

        with col1:
            st.metric("Avg Probability", f"{avg_prob*100:.1f}%")

        with col2:
            st.metric("High Probability Trades", high_prob_count)

        with col3:
            st.metric("Total Predictions", len(predictions))

    except Exception as e:
        st.error(f"❌ Error loading predictions: {e}")
        st.info("💡 Make sure the model is trained and available in models/")


def render_kol_future_performance():
    """Render KOL future performance predictions"""
    st.markdown("### 🔮 KOL Future Performance Prediction")

    models = check_model_availability()

    # KOL predictor is not available yet
    if not models['kol_predictor']:
        st.warning("⚠️ KOL Predictor needs more historical data (35+ days)")
        st.info("💡 The system is collecting data. KOL predictions will be available once there's enough history.")
        return

    # Placeholder for when KOL predictor is available
    st.info("🚧 KOL predictor coming soon!")


def render_model_metrics():
    """Render model performance metrics"""
    st.markdown("### 📉 Model Performance Metrics")

    models = check_model_availability()

    if not models['token_predictor']:
        st.warning("⚠️ No trained model found")
        return

    # Load model metadata
    try:
        import json
        from pathlib import Path

        metadata_path = Path('models/token_predictor_best_metadata.json')

        if not metadata_path.exists():
            st.warning("⚠️ Model metadata not found")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Display metrics
        st.markdown("#### Token Predictor Metrics")

        metrics = metadata.get('metrics', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            auc = metrics.get('auc', 0)
            st.metric("AUC-ROC", f"{auc:.4f}")

        with col2:
            precision = metrics.get('precision', 0)
            st.metric("Precision", f"{precision:.2%}")

        with col3:
            recall = metrics.get('recall', 0)
            st.metric("Recall", f"{recall:.2%}")

        with col4:
            f1 = metrics.get('f1', 0)
            st.metric("F1-Score", f"{f1:.2%}")

        # Training info
        st.markdown("#### 📊 Training Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            samples = metadata.get('total_samples', 0)
            st.metric("Training Samples", f"{samples:,}")

        with col2:
            epochs = metadata.get('epochs', 0)
            st.metric("Epochs Trained", epochs)

        with col3:
            pos_count = metadata.get('positive_samples', 0)
            st.metric("3x+ Positions", pos_count)

        # Show training progress if available
        if 'history' in metadata:
            history = metadata['history']

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Loss', 'Validation AUC')
            )

            epochs_range = list(range(1, len(history.get('train_loss', [])) + 1))

            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=history.get('train_loss', []),
                mode='lines',
                name='Train Loss',
                line=dict(color='#FF6B6B')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=history.get('val_auc', []),
                mode='lines',
                name='Val AUC',
                line=dict(color='#4ECDC4')
            ), row=1, col=2)

            fig.update_layout(
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading model metrics: {e}")


def render_feature_importance():
    """Render feature importance visualization"""
    st.markdown("### 🔍 Feature Importance")

    models = check_model_availability()

    if not models['token_predictor']:
        st.warning("⚠️ No trained model found")
        return

    # Feature descriptions
    features = [
        'DH Score',
        '3x+ Rate',
        'Win Rate',
        'Avg Hold Time',
        'Amount SOL',
        'Entry Hour',
        'Entry Day'
    ]

    # Placeholder feature importance (would come from model)
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03]

    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#4ECDC4',
            text=[f"{v:.1%}" for v in importance],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Feature Importance for Token Prediction",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature descriptions
    st.markdown("#### 📖 Feature Descriptions")

    descriptions = {
        'DH Score': "Diamond Hand Score - Overall KOL quality metric",
        '3x+ Rate': "Historical 3x+ success rate of the KOL",
        'Win Rate': "Historical win rate (percentage of profitable trades)",
        'Avg Hold Time': "Average time KOL holds positions (normalized)",
        'Amount SOL': "Amount of SOL invested (log normalized)",
        'Entry Hour': "Hour of day when trade was executed (0-23)",
        'Entry Day': "Day of week when trade was executed"
    }

    for feature, desc in descriptions.items():
        st.markdown(f"**{feature}**: {desc}")


def render_anomaly_detection():
    """Render anomaly detection for unusual patterns"""
    st.markdown("### ⚠️ Anomaly Detection")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Find anomalies using simple statistical rules
    anomalies = []

    # High score with low trades (potential one-hit wonder)
    one_hit_wonders = df[
        (df['diamond_hand_score'] > 50) &
        (df['total_trades'] < 5)
    ]

    # High win rate but no 3x+ (too conservative)
    conservative = df[
        (df['win_rate'] > 0.8) &
        (df['three_x_plus_rate'] == 0) &
        (df['total_trades'] > 10)
    ]

    # High PnL but low DH score (inconsistent)
    profitable_but_inconsistent = df[
        (df['total_pnl_sol'] > 10) &
        (df['diamond_hand_score'] < 30)
    ]

    if not one_hit_wonders.empty:
        st.markdown("#### 🎯 Potential One-Hit Wonders")
        st.info("💎 High score but very few trades - may be lucky")
        for _, row in one_hit_wonders.head(5).iterrows():
            st.write(f"• **{row['name']}**: Score {row['diamond_hand_score']:.1f} with only {int(row['total_trades'])} trade(s)")

    if not conservative.empty:
        st.markdown("#### 🛡️ Too Conservative")
        st.info("📊 High win rate but no 3x+ trades - missing opportunities")
        for _, row in conservative.head(5).iterrows():
            st.write(f"• **{row['name']}**: Win Rate {row['win_rate']*100:.1f}% but 0 3x+ trades")

    if not profitable_but_inconsistent.empty:
        st.markdown("#### 💰 Profitable but Inconsistent")
        st.info("📈 Good PnL but low DH score - volatility")
        for _, row in profitable_but_inconsistent.head(5).iterrows():
            st.write(f"• **{row['name']}**: PnL {row['total_pnl_sol']:.2f} SOL but Score {row['diamond_hand_score']:.1f}")

    if one_hit_wonders.empty and conservative.empty and profitable_but_inconsistent.empty:
        st.success("✅ No significant anomalies detected")


def render_what_if_simulator():
    """Render what-if analysis simulator"""
    st.markdown("### 🎲 What-If Analysis Simulator")

    models = check_model_availability()

    if not models['token_predictor']:
        st.warning("⚠️ Token Predictor model not trained yet")
        st.info("💡 Train the model first to use the simulator")
        return

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Simulator inputs
    col1, col2 = st.columns(2)

    with col1:
        kol_name = st.selectbox("Select KOL", df['name'].tolist(), key="whatif_kol")

    with col2:
        amount_sol = st.number_input(
            "Amount to Invest (SOL)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            key="whatif_amount"
        )

    entry_time = st.datetime_input(
        "Entry Time",
        value=datetime.now(),
        key="whatif_time"
    )

    if st.button("🎯 Calculate Probability", key="whatif_button"):
        if not kol_name:
            st.warning("⚠️ Please select a KOL")
            return

        kol_data = df[df['name'] == kol_name].iloc[0]

        # Load predictor and calculate
        try:
            from api.batch_predictor import BatchPredictor

            predictor = BatchPredictor()

            prob = predictor.predict_token_3x_probability(
                kol_id=kol_data['kol_id'],
                amount_sol=amount_sol,
                entry_time=entry_time
            )

            # Display results
            st.markdown("#### 📊 Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                # Probability gauge
                prob_percent = prob * 100

                if prob_percent >= 70:
                    color = "#FF6B6B"
                    label = "🔴 HIGH"
                elif prob_percent >= 40:
                    color = "#FFD700"
                    label = "🟡 MEDIUM"
                else:
                    color = "#4ECDC4"
                    label = "🟢 LOW"

                st.metric(
                    "3x+ Probability",
                    f"{prob_percent:.1f}%",
                )

                st.markdown(f"<h3 style='text-align: center; color: {color}; font-size: 4rem;'>{prob_percent:.1f}%</h3>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {color}; font-size: 1.5rem;'>{label}</p>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown("##### 📋 Trade Details")

                st.markdown(f"**KOL:** {kol_name}")
                st.markdown(f"**DH Score:** {kol_data['diamond_hand_score']:.1f}")
                st.markdown(f"**Win Rate:** {kol_data['win_rate']*100:.1f}%")
                st.markdown(f"**Investment:** {amount_sol} SOL")
                st.markdown(f"**Entry Time:** {entry_time.strftime('%Y-%m-%d %H:%M')}")

            # Risk assessment
            st.markdown("---")
            st.markdown("##### ⚠️ Risk Assessment")

            if prob_percent >= 70:
                st.warning("🔴 **High Risk / High Reward** - High probability but still risky!")
            elif prob_percent >= 40:
                st.info("🟡 **Medium Risk** - Moderate probability, use caution")
            else:
                st.success("🟢 **Lower Risk** - Lower probability, may be safer")

            # Disclaimer
            st.markdown("---")
            st.caption("⚠️ **Disclaimer**: This is a statistical prediction, not financial advice. "
                        "Always do your own research and never invest more than you can afford to lose.")

        except Exception as e:
            st.error(f"❌ Error calculating prediction: {e}")


def main():
    """Main entry point for ML predictions page"""
    st.title("🤖 ML Predictions")

    # Check model availability at top
    models = check_model_availability()

    if not models['models_available']:
        st.error("❌ No ML models available")
        st.markdown("### 🚀 Train Your First Model")

        st.markdown("To use ML predictions, you need to train the model first:")

        st.markdown("""
        **Steps:**
        1. Make sure you have enough closed positions in the database
        2. Run the training command:
        ```bash
        python run_continuous_trainer.py --model token --once --epochs 30
        ```

        3. Once training completes, the predictions will be available here
        """)

        # Show current data status
        data_manager = get_data_manager()
        stats = data_manager.get_database_stats()

        st.markdown("---")
        st.markdown("### 📊 Current Data Status")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Closed Positions", stats['total_positions'])

        with col2:
            # Estimate how many more positions needed
            more_needed = max(0, 100 - stats['total_positions'])
            if more_needed > 0:
                st.metric("Positions Needed for Training", more_needed)
            else:
                st.metric("Positions Ready", "✅ Enough")

        return

    # Show model status
    col1, col2 = st.columns(2)

    with col1:
        if models['token_predictor']:
            st.success("✅ Token Predictor: Available")
        else:
            st.warning("⏳ Token Predictor: Not trained")

    with col2:
        if models['kol_predictor']:
            st.success("✅ KOL Predictor: Available")
        else:
            st.info("🔄 KOL Predictor: Collecting data...")

    st.markdown("---")

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Token Probability",
        "🔮 KOL Performance",
        "📉 Model Metrics",
        "⚠️ Anomaly Detection",
        "🎲 What-If Simulator"
    ])

    with tab1:
        render_token_probability_table()

    with tab2:
        render_kol_future_performance()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            render_model_metrics()
        with col2:
            render_feature_importance()

    with tab4:
        render_anomaly_detection()

    with tab5:
        render_what_if_simulator()


if __name__ == "__main__":
    main()
