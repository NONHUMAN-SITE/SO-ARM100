from typing import Dict, Any
def get_weather(location: str, units: str) -> dict:
    """
    Retrieves current weather for the given location.

    Args:
        location (str): City and country, e.g., "Bogotá, Colombia".
        units (str): "celsius" or "fahrenheit".

    Returns:
        dict: Weather information like temperature and description.
    """
    # Example (you should replace this with a real API call)
    fake_weather_data = {
        "Bogotá, Colombia": {"celsius": 18, "fahrenheit": 64},
        "New York, USA": {"celsius": 22, "fahrenheit": 72},
    }

    temp = fake_weather_data.get(location, {"celsius": 20, "fahrenheit": 68})
    return {
        "location": location,
        "temperature": temp[units],
        "units": units,
        "description": "Partly cloudy"  # You can adjust this
    }