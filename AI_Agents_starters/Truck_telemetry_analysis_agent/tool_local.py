
# Minimal example – replace S3 with local file for testing
import pandas as pd
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Simulate reading data (replace with real S3)
@tool
def read_local_data(file_path: str) -> pd.DataFrame:
    """Reads a local Parquet file and returns a DataFrame."""
    return pd.read_parquet(file_path)

@tool
def compute_metrics(df: pd.DataFrame) -> dict:
    # Convert timestamp to datetime if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Metrics
    result = {}
    for truck in df['truck_id'].unique():
        truck_df = df[df['truck_id'] == truck]
        metrics = {
            'avg_speed_kmh': truck_df['speed'].mean(),
            'avg_fuel_consumption': truck_df['fuel_consumption'].mean(),
            'total_distance_km': (truck_df['speed'] * (truck_df['timestamp'].diff().dt.total_seconds() / 3600)).sum(),
            'idle_time_minutes': (truck_df[truck_df['speed'] == 0].shape[0] * (
                        truck_df['timestamp'].diff().dt.total_seconds().median() / 60)).sum(),
            'max_speed': truck_df['speed'].max()
        }
        result[truck] = metrics
    return result

@tool
def generate_report(metrics: dict) -> str:
    """Takes metrics dict and returns a formatted text report."""
    report_lines = ["Truck Analytics Report", "===================="]
    for truck, m in metrics.items():
        report_lines.append(f"Truck {truck}:")
        report_lines.append(f"  Avg Speed: {m['avg_speed_kmh']:.1f} km/h")
        report_lines.append(f"  Avg Fuel: {m['avg_fuel_consumption']:.1f} L/100km")
        report_lines.append(f"  Total Distance: {m['total_distance_km']:.0f} km")
        report_lines.append(f"  Idle Time: {m['idle_time_minutes']:.0f} min")
        report_lines.append(f"  Max Speed: {m['max_speed']:.0f} km/h")
        report_lines.append("")
    return "\n".join(report_lines)


tools = [read_local_data, compute_metrics, generate_report]
# ... rest of agent setup ...