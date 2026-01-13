"""
Summaries Page - Display automated run summaries

Shows:
- Latest summary
- Historical summaries
- Performance trends
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import json

from dashboard.core.summary_generator import get_summary_generator
from dashboard.styles.theme import render_theme


def render_latest_summary():
    """Render the latest summary"""
    st.subheader("📊 Latest Run Summary")

    # Load latest summary
    latest_path = Path('data/summaries/latest_summary.json')

    if not latest_path.exists():
        st.info("⏳ No summaries generated yet. The summary scheduler will generate the first summary in 15 minutes.")
        return

    try:
        with open(latest_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        # Get generator for markdown conversion
        generator = get_summary_generator()
        markdown = generator.get_summary_markdown(summary)

        # Display markdown
        st.markdown(markdown)

    except Exception as e:
        st.error(f"Error loading summary: {e}")


def render_historical_summaries():
    """Render historical summaries"""
    st.subheader("📜 Historical Summaries")

    summaries_dir = Path('data/summaries')

    if not summaries_dir.exists():
        st.warning("No summaries directory found")
        return

    # Get all summary files
    summary_files = sorted(
        summaries_dir.glob('summary_*.json'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not summary_files:
        st.info("No historical summaries available yet.")
        st.caption("💡 The scheduler generates a new summary every 15 minutes. Check back soon!")
        return

    # Display information
    st.write(f"**Total summaries:** {len(summary_files)}")
    st.write(f"**Showing last {min(20, len(summary_files))} summaries:**")
    st.markdown("---")

    # Display summaries in a clean format
    for i, filepath in enumerate(summary_files[:20]):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            # Extract timestamp
            timestamp = summary.get('timestamp', 'Unknown')
            dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now()

            # Get stats
            run_stats = summary.get('run_stats', {})
            kol_stats = summary.get('kol_stats', {})

            recent_trades = run_stats.get('recent_trades_last_15min', 0)
            total_kols = run_stats.get('total_kols', 0)
            total_trades = run_stats.get('total_trades', 0)
            avg_score = kol_stats.get('avg_diamond_hand_score', 0)
            win_rate = kol_stats.get('avg_win_rate_percentage', 0)

            # Create expander with better title
            with st.expander(f"📅 **{dt.strftime('%Y-%m-%d %H:%M')}** - {recent_trades} new trades"):
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("KOLs", f"{total_kols:,}")

                with col2:
                    st.metric("Total Trades", f"{total_trades:,}")

                with col3:
                    st.metric("Avg Score", f"{avg_score:.1f}")

                with col4:
                    st.metric("Win Rate", f"{win_rate:.1f}%")

                with col5:
                    st.metric("New (15m)", f"{recent_trades}")

                # Show top 3 performers for this summary
                top_performers = summary.get('top_performers', {}).get('top_10', [])
                if top_performers:
                    st.markdown("**🏆 Top 3 Performers:**")
                    for j, perf in enumerate(top_performers[:3]):
                        st.markdown(f"{j+1}. **{perf['name']}** - Score: {perf['diamond_hand_score']:.1f}")

        except Exception as e:
            st.error(f"❌ Error loading {filepath.name}: {e}")
            import traceback
            st.code(traceback.format_exc())


def render_scheduler_status():
    """Render summary scheduler status"""
    st.subheader("⚙️ Scheduler Status")

    # Check if scheduler is running
    import psutil

    scheduler_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'run_summary_scheduler.py' in ' '.join(cmdline):
                scheduler_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if scheduler_running:
        st.success("✅ Summary scheduler is running")
        st.info("📊 Summaries are generated every 15 minutes")
    else:
        st.warning("⚠️ Summary scheduler is not running")
        st.info("To start the scheduler, run: `python run_summary_scheduler.py`")


def main():
    """Main entry point for summaries page"""
    st.title("📊 Automated Summaries")

    # Scheduler status
    render_scheduler_status()

    st.markdown("---")

    # Latest summary
    render_latest_summary()

    st.markdown("---")

    # Historical summaries
    render_historical_summaries()


if __name__ == "__main__":
    main()
