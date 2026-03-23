# Example list of S3 keys (for demonstration)
s3_bucket = "your-truck-data-bucket"
date_range = ["2025-03-15", "2025-03-16", "2025-03-17"]  # replace with actual keys
s3_keys = [f"processed/truck_data_{date}.parquet" for date in date_range]

all_reports = []

for key in s3_keys:
    # We instruct the agent to analyze this file
    instruction = f"""
    Read the truck data from S3 bucket '{s3_bucket}' with key '{key}'.
    Compute metrics for each truck and generate a summary report.
    """
    response = agent_executor.invoke({"input": instruction})
    all_reports.append(f"Report for {key}:\n{response['output']}")

# Save all reports to a file or S3
final_report = "\n\n".join(all_reports)
print(final_report)