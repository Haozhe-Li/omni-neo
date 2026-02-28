from pyowm import OWM
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file
import os
from core.utils.redis_cache import l1cache


owm = OWM(api_key=os.getenv("OPENWEATHERMAP_API_KEY"))
mgr = owm.weather_manager()


@l1cache(ttl=3600)
def get_weather(location: str) -> dict:
    """Get current weather for a location. location (str): The location to get the weather for. MUST be in English."""
    try:
        observation = mgr.weather_at_place(location)
        weather = observation.weather
        res = weather.to_dict()
        res["location"] = location
        return res
    except Exception as e:
        return {"error": f"Failed to get weather data for {location}: {str(e)}"}
