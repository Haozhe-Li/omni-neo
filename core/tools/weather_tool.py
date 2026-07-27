from pyowm import OWM
import dotenv
import os
from datetime import datetime, timezone

dotenv.load_dotenv()
from core.utils.redis_cache import l1cache
from core.utils.citations import register_citation


owm = OWM(api_key=os.getenv("OPENWEATHERMAP_API_KEY"))
mgr = owm.weather_manager()


def _weather_source_url(lat: float, lon: float, kind: str) -> str:
    """Weather readings have no natural webpage of their own — this points at
    OWM's public map centered on the queried coordinates, close enough to a
    real source link that the frontend can render it like any other citation.
    `kind` ("current"/"forecast") keeps the two tools' citations from deduping
    into one another when both are called for the same place in one thread —
    otherwise a later forecast call would silently reuse (and never update)
    the citation content an earlier current-weather call registered."""
    return f"https://openweathermap.org/weathermap?zoom=10&lat={lat:.4f}&lon={lon:.4f}#{kind}"


def _current_summary_text(weather) -> str:
    """One-line human-readable summary of a PyOWM Weather object, used as
    citation content (Celsius — `to_dict()` on the same object stays in
    Kelvin for backward compatibility with existing callers)."""
    temp_c = weather.temperature('celsius')
    wind = weather.wind('meters_sec')
    return (
        f"{weather.detailed_status or weather.status}, "
        f"{round(temp_c.get('temp', 0))}°C "
        f"(feels like {round(temp_c.get('feels_like', temp_c.get('temp', 0)))}°C), "
        f"humidity {weather.humidity}%, wind {round(wind.get('speed', 0), 1)} m/s."
    )


# @l1cache(ttl=3600)
def get_weather(location: str) -> dict:
    """Get current weather for a location. location (str): The location to get the weather for. MUST be in English."""
    try:
        observation = mgr.weather_at_place(location)
        weather = observation.weather
        res = weather.to_dict()
        res["location"] = location

        n = register_citation(
            title=f"Current weather for {location}",
            url=_weather_source_url(observation.location.lat, observation.location.lon, "current"),
            content=f"Current weather in {location}: {_current_summary_text(weather)}",
        )
        if n is not None:
            res["n"] = n
        return res
    except Exception as e:
        return {"error": f"Failed to get weather data for {location}: {str(e)}"}


def _format_forecast_slot(w) -> dict:
    """Format a single PyOWM Weather forecast object (hourly granularity) into a clean dict."""
    ref_dt = w.reference_time('date')
    temp = w.temperature('celsius')
    wind = w.wind('meters_sec')
    rain = w.rain or {}
    snow = w.snow or {}

    slot: dict = {
        "datetime_utc": ref_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "time": ref_dt.strftime('%H:%M'),
        "date": ref_dt.strftime('%Y-%m-%d'),
        "temp_c": round(temp.get('temp', 0), 1),
        "temp_max_c": round(temp.get('temp_max', temp.get('temp', 0)), 1),
        "temp_min_c": round(temp.get('temp_min', temp.get('temp', 0)), 1),
        "feels_like_c": round(temp.get('feels_like', temp.get('temp', 0)), 1),
        "status": w.status or '',
        "detailed_status": w.detailed_status or '',
        "humidity": w.humidity,
        "wind_speed": round(wind.get('speed', 0), 1),
        "wind_deg": wind.get('deg'),
        "weather_code": w.weather_code,
        "icon": w.weather_icon_name or '',
        "pop": 0,
    }

    if w.precipitation_probability is not None:
        slot["pop"] = round(w.precipitation_probability * 100)

    rain_mm = rain.get('1h') or rain.get('3h') or rain.get('all')
    if rain_mm:
        slot["rain_mm"] = round(float(rain_mm), 1)

    snow_mm = snow.get('1h') or snow.get('3h') or snow.get('all')
    if snow_mm:
        slot["snow_mm"] = round(float(snow_mm), 1)

    return slot


def _format_daily_slot(w) -> dict:
    """Format a single PyOWM daily-forecast Weather object (One Call API 3.0,
    which reports day/night/eve/morn temps directly rather than 3-hour slots
    that need aggregating)."""
    ref_dt = w.reference_time('date')
    temp = w.temperature('celsius')
    wind = w.wind('meters_sec')
    rain = w.rain or {}
    snow = w.snow or {}

    slot: dict = {
        "date": ref_dt.strftime('%Y-%m-%d'),
        "temp_day_c": round(temp.get('day', 0), 1),
        "temp_night_c": round(temp.get('night', 0), 1),
        "temp_max_c": round(temp.get('max', temp.get('day', 0)), 1),
        "temp_min_c": round(temp.get('min', temp.get('day', 0)), 1),
        "status": w.status or '',
        "detailed_status": w.detailed_status or '',
        "humidity": w.humidity,
        "wind_speed": round(wind.get('speed', 0), 1),
        "wind_deg": wind.get('deg'),
        "weather_code": w.weather_code,
        "icon": w.weather_icon_name or '',
        "pop": round((w.precipitation_probability or 0) * 100),
    }

    rain_mm = rain.get('all')
    if rain_mm:
        slot["rain_mm"] = round(float(rain_mm), 1)

    snow_mm = snow.get('all')
    if snow_mm:
        slot["snow_mm"] = round(float(snow_mm), 1)

    return slot


def _forecast_citation_content(location: str, current_weather, daily: list[dict]) -> str:
    parts = [f"Current weather in {location}: {_current_summary_text(current_weather)}"]
    for day in daily[:8]:
        parts.append(
            f"{day['date']}: {day['detailed_status']}, {day['temp_min_c']}-{day['temp_max_c']}°C."
        )
    return " ".join(parts)


# @l1cache(ttl=3600)
def get_weather_forecast(location: str) -> dict:
    """Get current weather PLUS today's hourly forecast and a daily outlook for the
    upcoming week for a location. Use this when the user asks about weather forecasts,
    tomorrow's weather, specific hours today, next week, or any other upcoming conditions.
    location (str): City/place name, MUST be in English.
    Returns current conditions, remaining hourly slots for today, and `daily_forecast` —
    a list of daily summaries from today through as many days ahead as the provider
    supplies (typically 8 days total, i.e. today + 1 week). If the user asks about a day
    beyond what `daily_forecast` covers, say the forecast doesn't reach that far instead
    of guessing."""
    try:
        obs = mgr.weather_at_place(location)
        lat, lon = obs.location.lat, obs.location.lon

        oc = mgr.one_call(lat=lat, lon=lon)

        current = oc.current.to_dict()
        current["location"] = location

        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        today_hourly = [
            _format_forecast_slot(w)
            for w in (oc.forecast_hourly or [])
            if w.reference_time('date').date() == today and w.reference_time('date') >= now_utc
        ]

        daily_forecast = [_format_daily_slot(w) for w in (oc.forecast_daily or [])]

        result: dict = {
            "location": location,
            "current": current,
            "today_hourly": today_hourly,
            "daily_forecast": daily_forecast,
        }
        # Kept for backward compatibility with earlier callers expecting these two keys.
        if len(daily_forecast) > 1:
            result["tomorrow"] = daily_forecast[1]
        if len(daily_forecast) > 2:
            result["day_after_tomorrow"] = daily_forecast[2]

        n = register_citation(
            title=f"Weather forecast for {location}",
            url=_weather_source_url(lat, lon, "forecast"),
            content=_forecast_citation_content(location, oc.current, daily_forecast),
        )
        if n is not None:
            result["n"] = n

        return result
    except Exception as e:
        return {"error": f"Failed to get weather forecast for {location}: {str(e)}"}
