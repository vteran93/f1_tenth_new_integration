#!/usr/bin/env python3
"""
Interactive Analysis Dashboard for F1TENTH RL Training Results
Provides comprehensive visualization and analysis of training metrics,
benchmark data, and model performance.
"""

import f1tenth_benchmarks
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys
from typing import List, Optional
import logging

# Configure page
st.set_page_config(
    page_title="F1TENTH RL Analysis Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use local f1tenth_benchmarks module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1TenthAnalysisDashboard:
    """Main dashboard class for F1TENTH analysis."""

    def __init__(self):
        self.results_dir = Path("./examples/ray_results")
        self.benchmark_dir = Path("./Logs")
        self.models_dir = Path("./examples/models")

    def load_training_results(self, experiment_path: str) -> Optional[pd.DataFrame]:
        """Load training results from Ray RLLib experiment."""
        try:
            experiment_dir = Path(experiment_path)
            if not experiment_dir.exists():
                st.error(f"Experiment directory not found: {experiment_path}")
                return None

            # Look for progress.csv or result.json files
            progress_file = experiment_dir / "progress.csv"
            if progress_file.exists():
                return pd.read_csv(progress_file)

            # Alternative: look for result.json files
            result_files = list(experiment_dir.rglob("result.json"))
            if result_files:
                results = []
                for file in result_files:
                    with open(file, 'r') as f:
                        for line in f:
                            results.append(json.loads(line))
                return pd.DataFrame(results)

            st.warning(f"No training results found in {experiment_path}")
            return None

        except Exception as e:
            st.error(f"Error loading training results: {e}")
            return None

    def load_benchmark_data(self, experiment_name: str) -> Optional[pd.DataFrame]:
        """Load benchmark data for analysis."""
        try:
            benchmark_path = self.benchmark_dir / experiment_name
            if not benchmark_path.exists():
                return None

            # Look for benchmark data files
            data_files = list(benchmark_path.rglob("*.csv"))
            if not data_files:
                return None

            # Combine all CSV files
            dfs = []
            for file in data_files:
                df = pd.read_csv(file)
                df['source_file'] = file.name
                dfs.append(df)

            return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            st.error(f"Error loading benchmark data: {e}")
            return None

    def plot_training_metrics(self, df: pd.DataFrame) -> None:
        """Create comprehensive training metrics plots."""
        if df is None or df.empty:
            st.warning("No training data available for plotting")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Episode Rewards", "Episode Length", "Learning Rate", "Loss"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Episode rewards
        if 'episode_reward_mean' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['episode_reward_mean'],
                           name="Mean Reward", line=dict(color='blue')),
                row=1, col=1
            )

        if 'episode_reward_max' in df.columns and 'episode_reward_min' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['episode_reward_max'],
                           name="Max Reward", line=dict(color='green', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['episode_reward_min'],
                           name="Min Reward", line=dict(color='red', dash='dash')),
                row=1, col=1
            )

        # Episode length
        if 'episode_len_mean' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['episode_len_mean'],
                           name="Mean Length", line=dict(color='orange')),
                row=1, col=2
            )

        # Learning rate
        if 'info/learner/default_policy/cur_lr' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['info/learner/default_policy/cur_lr'],
                           name="Learning Rate", line=dict(color='purple')),
                row=2, col=1
            )

        # Loss
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        if loss_cols:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[loss_cols[0]],
                           name="Loss", line=dict(color='red')),
                row=2, col=2
            )

        fig.update_layout(
            title="Training Metrics Overview",
            showlegend=True,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_benchmark_analysis(self, df: pd.DataFrame) -> None:
        """Create benchmark analysis plots."""
        if df is None or df.empty:
            st.warning("No benchmark data available for plotting")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Lap time distribution
            if 'lap_time' in df.columns:
                fig = px.histogram(df, x='lap_time', title="Lap Time Distribution",
                                   nbins=30, marginal="box")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Progress over time
            if 'progress' in df.columns and 'episode' in df.columns:
                fig = px.scatter(df, x='episode', y='progress',
                                 title="Progress Over Episodes",
                                 trendline="lowess")
                st.plotly_chart(fig, use_container_width=True)

        # Speed and steering analysis
        if 'speed' in df.columns and 'steering' in df.columns:
            fig = px.scatter(df, x='speed', y='steering',
                             title="Speed vs Steering Analysis",
                             opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

    def show_model_comparison(self, experiments: List[str]) -> None:
        """Compare multiple models/experiments."""
        if len(experiments) < 2:
            st.warning("Select at least 2 experiments for comparison")
            return

        comparison_data = []
        for exp in experiments:
            df = self.load_training_results(f"./examples/ray_results/{exp}")
            if df is not None and not df.empty:
                # Get final metrics
                final_metrics = {
                    'experiment': exp,
                    'final_reward': df['episode_reward_mean'].iloc[-1] if 'episode_reward_mean' in df.columns else None,
                    'max_reward': df['episode_reward_max'].max() if 'episode_reward_max' in df.columns else None,
                    'final_episode_length': df['episode_len_mean'].iloc[-1] if 'episode_len_mean' in df.columns else None,
                    'training_iterations': len(df),
                }
                comparison_data.append(final_metrics)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)

            # Create comparison plots
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Final Reward", "Max Reward", "Final Episode Length")
            )

            fig.add_trace(
                go.Bar(x=comparison_df['experiment'], y=comparison_df['final_reward'],
                       name="Final Reward"),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(x=comparison_df['experiment'], y=comparison_df['max_reward'],
                       name="Max Reward"),
                row=1, col=2
            )

            fig.add_trace(
                go.Bar(x=comparison_df['experiment'], y=comparison_df['final_episode_length'],
                       name="Final Episode Length"),
                row=1, col=3
            )

            fig.update_layout(title="Model Comparison", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Show comparison table
            st.subheader("Detailed Comparison")
            st.dataframe(comparison_df)

    def run_dashboard(self):
        """Main dashboard interface."""
        st.title("🏎️ F1TENTH RL Analysis Dashboard")
        st.markdown("---")

        # Sidebar
        st.sidebar.header("Analysis Options")

        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Training Metrics", "Benchmark Analysis", "Model Comparison", "Live Monitoring"]
        )

        if analysis_type == "Training Metrics":
            st.header("Training Metrics Analysis")

            # Get available experiments
            experiments = []
            if self.results_dir.exists():
                experiments = [d.name for d in self.results_dir.iterdir() if d.is_dir()]

            if experiments:
                selected_exp = st.selectbox("Select Experiment", experiments)

                if selected_exp:
                    df = self.load_training_results(f"./examples/ray_results/{selected_exp}")
                    if df is not None:
                        self.plot_training_metrics(df)

                        # Show raw data
                        if st.checkbox("Show Raw Data"):
                            st.subheader("Raw Training Data")
                            st.dataframe(df)
            else:
                st.warning("No training experiments found. Run some training first!")

        elif analysis_type == "Benchmark Analysis":
            st.header("Benchmark Data Analysis")

            # Get available benchmark data
            benchmark_experiments = []
            if self.benchmark_dir.exists():
                benchmark_experiments = [d.name for d in self.benchmark_dir.iterdir() if d.is_dir()]

            if benchmark_experiments:
                selected_benchmark = st.selectbox("Select Benchmark Experiment", benchmark_experiments)

                if selected_benchmark:
                    df = self.load_benchmark_data(selected_benchmark)
                    if df is not None:
                        self.plot_benchmark_analysis(df)

                        if st.checkbox("Show Benchmark Raw Data"):
                            st.subheader("Raw Benchmark Data")
                            st.dataframe(df)
            else:
                st.warning("No benchmark data found. Run some experiments with benchmark collection first!")

        elif analysis_type == "Model Comparison":
            st.header("Model Comparison")

            experiments = []
            if self.results_dir.exists():
                experiments = [d.name for d in self.results_dir.iterdir() if d.is_dir()]

            if experiments:
                selected_experiments = st.multiselect("Select Experiments to Compare", experiments)

                if len(selected_experiments) >= 2:
                    self.show_model_comparison(selected_experiments)
            else:
                st.warning("No experiments found for comparison!")

        elif analysis_type == "Live Monitoring":
            st.header("Live Training Monitoring")
            st.info("🚧 Live monitoring feature coming soon!")
            st.markdown("""
            This feature will provide:
            - Real-time training metrics
            - Live plots updating during training
            - Performance alerts and notifications
            - Resource utilization monitoring
            """)


def main():
    """Main function to run the dashboard."""
    dashboard = F1TenthAnalysisDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
