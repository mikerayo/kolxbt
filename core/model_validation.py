"""
Model Validation Module for KOL Tracker ML

Valida la precisión de las predicciones del modelo ML:
- Accuracy, Precision, Recall, F1 Score
- ROC AUC, Confusion Matrix
- Calibration curves
- Prediction confidence analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from core.database import db, ClosedPosition
from core.ml_models import MLPipeline
from core.feature_engineering import KOLFeatures
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Métricas de validación del modelo"""
    model_name: str
    validation_date: datetime

    # Métricas de clasificación
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Métricas de ranking
    roc_auc: float
    avg_precision: float

    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Calibración
    expected_positive_rate: float
    actual_positive_rate: float
    calibration_error: float

    # Por nivel de confianza
    confidence_accuracy: Dict[str, float]

    # Reporte completo
    classification_report: str


class ModelValidator:
    """
    Valida las predicciones del modelo ML contra resultados reales
    """

    def __init__(self):
        self.session = db.get_session()
        self.ml_pipeline = MLPipeline()

    def validate_predictions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_trades_per_kol: int = 5
    ) -> ValidationMetrics:
        """
        Valida predicciones del modelo vs resultados reales

        Args:
            start_date: Fecha de inicio de validación
            end_date: Fecha de fin de validación
            min_trades_per_kol: Mínimo de trades por KOL para incluir

        Returns:
            ValidationMetrics con todas las métricas
        """
        logger.info("Starting model validation...")

        # Obtener posiciones cerradas (ground truth)
        positions = self._get_validation_positions(
            start_date,
            end_date,
            min_trades_per_kol
        )

        if not positions:
            logger.warning("No positions found for validation")
            return self._empty_validation_metrics()

        # Obtener features para cada posición
        X, y_true = self._prepare_validation_data(positions)

        # Hacer predicciones
        y_pred_proba = self.ml_pipeline.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0

        # Average Precision
        from sklearn.metrics import average_precision_score
        avg_prec = average_precision_score(y_true, y_pred_proba)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calibración
        expected_pos_rate = y_pred_proba.mean()
        actual_pos_rate = y_true.mean()
        calibration_error = abs(expected_pos_rate - actual_pos_rate)

        # Accuracy por nivel de confianza
        confidence_accuracy = self._calculate_confidence_accuracy(
            y_true, y_pred_proba
        )

        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            target_names=['Not 3x+', '3x+'],
            zero_division=0
        )

        return ValidationMetrics(
            model_name="Diamond Hand Predictor",
            validation_date=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            avg_precision=avg_prec,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            expected_positive_rate=expected_pos_rate,
            actual_positive_rate=actual_pos_rate,
            calibration_error=calibration_error,
            confidence_accuracy=confidence_accuracy,
            classification_report=class_report
        )

    def backtest_model_performance_over_time(
        self,
        window_days: int = 30,
        min_trades_per_window: int = 10
    ) -> pd.DataFrame:
        """
        Evalúa performance del modelo en ventanas de tiempo (rolling validation)

        Args:
            window_days: Días por ventana
            min_trades_per_window: Mínimo de trades por ventana

        Returns:
            DataFrame con métricas por ventana
        """
        logger.info(f"Running rolling validation with {window_days}-day windows...")

        # Obtener todas las posiciones
        positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.pnl_multiple.isnot(None)
        ).order_by(ClosedPosition.exit_time).all()

        if not positions:
            return pd.DataFrame()

        # Crear ventanas de tiempo
        start_date = positions[0].exit_time
        end_date = positions[-1].exit_time

        windows = []
        current_date = start_date

        while current_date <= end_date:
            window_end = current_date + timedelta(days=window_days)

            # Filtrar posiciones en esta ventana
            window_positions = [
                p for p in positions
                if current_date <= p.exit_time < window_end
            ]

            if len(window_positions) >= min_trades_per_window:
                # Validar en esta ventana
                X, y_true = self._prepare_validation_data(window_positions)
                y_pred_proba = self.ml_pipeline.predict_proba(X)
                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Calcular métricas
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                windows.append({
                    'window_start': current_date,
                    'window_end': window_end,
                    'n_trades': len(window_positions),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

            current_date = window_end

        return pd.DataFrame(windows)

    def get_top_predictions_analysis(
        self,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Analiza las N mejores predicciones del modelo

        Args:
            top_n: Número de mejores predicciones a analizar

        Returns:
            DataFrame con análisis
        """
        logger.info(f"Analyzing top {top_n} predictions...")

        # Obtener todas las posiciones con predicciones
        positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.pnl_multiple.isnot(None)
        ).all()

        if not positions:
            return pd.DataFrame()

        # Preparar datos
        X, y_true = self._prepare_validation_data(positions)
        y_pred_proba = self.ml_pipeline.predict_proba(X)

        # Crear DataFrame con predicciones
        results_df = pd.DataFrame({
            'position_index': range(len(positions)),
            'predicted_probability': y_pred_proba,
            'actual_result': y_true,
            'pnl_multiple': [p.pnl_multiple for p in positions if p.pnl_multiple else 0],
            'exit_time': [p.exit_time for p in positions]
        })

        # Ordenar por probabilidad predicha
        results_df = results_df.sort_values('predicted_probability', ascending=False)

        # Analizar top N
        top_predictions = results_df.head(top_n)

        analysis = {
            'metric': [
                'Top N Predictions - Actual 3x+ Rate',
                'Top N Predictions - Avg Multiple',
                'All Predictions - Actual 3x+ Rate',
                'All Predictions - Avg Multiple',
                'Improvement (3x+ Rate)',
                'Improvement (Avg Multiple)'
            ],
            'value': [
                top_predictions['actual_result'].mean(),
                top_predictions['pnl_multiple'].mean(),
                results_df['actual_result'].mean(),
                results_df['pnl_multiple'].mean(),
                (top_predictions['actual_result'].mean() - results_df['actual_result'].mean()) * 100,
                top_predictions['pnl_multiple'].mean() - results_df['pnl_multiple'].mean()
            ]
        }

        return pd.DataFrame(analysis)

    def _get_validation_positions(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        min_trades_per_kol: int
    ) -> List[ClosedPosition]:
        """Obtiene posiciones para validación"""
        query = self.session.query(ClosedPosition).filter(
            ClosedPosition.pnl_multiple.isnot(None)
        )

        if start_date:
            query = query.filter(ClosedPosition.exit_time >= start_date)
        if end_date:
            query = query.filter(ClosedPosition.exit_time <= end_date)

        positions = query.all()

        # Filtrar por trades mínimos (se podría hacer más complejo)
        return positions

    def _prepare_validation_data(
        self,
        positions: List[ClosedPosition]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara X, y para validación"""
        # Obtener KOLs únicos
        from sqlalchemy import func
        kol_ids = list(set([p.kol_id for p in positions]))

        # Calcular features para cada KOL
        feature_engine = KOLFeatures()
        features_list = []

        for kol_id in kol_ids:
            kol = self.session.query(db.KOL).filter(db.KOL.id == kol_id).first()
            if kol:
                features = feature_engine.calculate_kol_features(self.session, kol)
                features_list.append(features)

        # Para cada posición, necesitamos sus features
        # Por simplicidad, usamos los features del KOL en ese momento
        # (Idealmente usaríamos features en el momento del trade)

        # Crear X matrix
        X = []
        y_true = []

        for pos in positions:
            # Buscar features del KOL
            kol_features = next(
                (f for f in features_list if f.get('kol_id') == pos.kol_id),
                None
            )

            if kol_features:
                # Crear feature vector (seleccionando features relevantes)
                feature_vector = [
                    kol_features.get('three_x_rate', 0),
                    kol_features.get('win_rate', 0),
                    kol_features.get('avg_hold_time_hours', 0) / 24,  # normalizar
                    kol_features.get('consistency_score', 0),
                    kol_features.get('total_trades', 0) / 100,  # normalizar
                ]

                X.append(feature_vector)

                # Label: 1 si fue 3x+, 0 si no
                y_true.append(1 if pos.pnl_multiple >= 3.0 else 0)

        return np.array(X), np.array(y_true)

    def _calculate_confidence_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula accuracy por nivel de confianza del modelo

        Divide las predicciones en bins de confianza:
        - Very High (80-100%)
        - High (60-80%)
        - Medium (40-60%)
        - Low (20-40%)
        - Very Low (0-20%)
        """
        confidence_bins = {
            'Very High (80-100%)': (0.8, 1.0),
            'High (60-80%)': (0.6, 0.8),
            'Medium (40-60%)': (0.4, 0.6),
            'Low (20-40%)': (0.2, 0.4),
            'Very Low (0-20%)': (0.0, 0.2)
        }

        confidence_accuracy = {}

        for bin_name, (lower, upper) in confidence_bins.items():
            # Filtrar predicciones en este bin
            mask = (y_pred_proba >= lower) & (y_pred_proba < upper)

            if mask.sum() > 0:
                bin_accuracy = accuracy_score(
                    y_true[mask],
                    (y_pred_proba[mask] >= 0.5).astype(int)
                )
                confidence_accuracy[bin_name] = bin_accuracy
            else:
                confidence_accuracy[bin_name] = 0.0

        return confidence_accuracy

    def _empty_validation_metrics(self) -> ValidationMetrics:
        """Retorna métricas vacías"""
        return ValidationMetrics(
            model_name="Diamond Hand Predictor",
            validation_date=datetime.now(),
            accuracy=0,
            precision=0,
            recall=0,
            f1_score=0,
            roc_auc=0,
            avg_precision=0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            expected_positive_rate=0,
            actual_positive_rate=0,
            calibration_error=0,
            confidence_accuracy={},
            classification_report="No data available"
        )


def validate_model() -> ValidationMetrics:
    """
    Función helper para validar el modelo

    Returns:
        ValidationMetrics
    """
    validator = ModelValidator()
    return validator.validate_predictions()
