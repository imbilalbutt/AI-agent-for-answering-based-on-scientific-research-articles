from datetime import datetime
from langchain.tools import tool

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time in the specified format."""
    return datetime.now().strftime(format)