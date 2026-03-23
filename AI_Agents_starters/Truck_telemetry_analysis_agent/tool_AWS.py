import boto3
import pandas as pd
from io import BytesIO
from langchain.tools import tool


# ---------- Tool 1: Read data from S3 ----------
@tool
def read_truck_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Reads a Parquet file from S3 and returns a pandas DataFrame."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    return df


# ---------- Tool 2: Compute truck metrics ----------
@tool
def compute_truck_metrics(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns:
    - truck_id
    - timestamp (datetime)
    - speed (km/h)
    - fuel_consumption (L/100km)
    - latitude, longitude
    - engine_status (on/off)
    Returns aggregated metrics per truck.
    """
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


# ---------- Tool 3: Generate a human-readable report ----------
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

