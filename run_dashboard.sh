#!/bin/bash
# Dashboard launcher

echo "Starting KOL Tracker Dashboard..."
echo "Open your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
streamlit run dashboard_v2.py
