from pyowm import OWM
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
import os
from functools import lru_cache


owm = OWM(api_key=os.getenv("OPENWEATHERMAP_API_KEY"))
mgr = owm.weather_manager()


@lru_cache(maxsize=128)
def get_weather(location: str) -> dict:
    try:
        observation = mgr.weather_at_place(location)
        weather = observation.weather
        res = weather.to_dict()
        res["location"] = location
        return res
    except Exception as e:
        return {"error": f"Failed to get weather data for {location}: {str(e)}"}

