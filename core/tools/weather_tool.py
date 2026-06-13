from pyowm import OWM
import dotenv
import os
from datetime import datetime, timezone, timedelta
from collections import Counter

dotenv.load_dotenv()
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


def _format_forecast_slot(w) -> dict:
    """Format a single PyOWM Weather forecast object into a clean dict."""
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

    rain_mm = rain.get('3h') or rain.get('1h') or rain.get('all')
    if rain_mm:
        slot["rain_mm"] = round(float(rain_mm), 1)

    snow_mm = snow.get('3h') or snow.get('1h') or snow.get('all')
    if snow_mm:
        slot["snow_mm"] = round(float(snow_mm), 1)

    return slot


def _aggregate_daily(slots: list[dict]) -> dict:
    """Aggregate 3-hour forecast slots into a single daily summary."""
    if not slots:
        return {}

    temp_maxes = [s["temp_max_c"] for s in slots]
    temp_mins = [s["temp_min_c"] for s in slots]
    humidities = [s["humidity"] for s in slots if s.get("humidity") is not None]
    wind_speeds = [s["wind_speed"] for s in slots if s.get("wind_speed") is not None]
    pops = [s.get("pop", 0) for s in slots]

    dominant_status = Counter(s["status"] for s in slots).most_common(1)[0][0] if slots else ""
    dominant_detailed = Counter(s["detailed_status"] for s in slots).most_common(1)[0][0] if slots else ""

    # Prefer a midday slot's icon (most representative)
    midday = next((s for s in slots if '11:00' <= s["time"] <= '14:00'), slots[len(slots) // 2])

    return {
        "date": slots[0]["date"],
        "temp_max_c": round(max(temp_maxes)) if temp_maxes else None,
        "temp_min_c": round(min(temp_mins)) if temp_mins else None,
        "status": dominant_status,
        "detailed_status": dominant_detailed,
        "humidity": round(sum(humidities) / len(humidities)) if humidities else None,
        "wind_speed": round(sum(wind_speeds) / len(wind_speeds), 1) if wind_speeds else None,
        "pop": max(pops) if pops else 0,
        "icon": midday.get("icon", ""),
        "hourly_slots": slots,
    }


@l1cache(ttl=3600)
def get_weather_forecast(location: str) -> dict:
    """Get current weather PLUS today's hourly forecast and a 2-day outlook for a location.
    Use this when the user asks about weather forecasts, tomorrow's weather, specific hours
    today, or upcoming conditions. location (str): City/place name, MUST be in English.
    Returns current conditions, remaining 3-hour slots for today, and daily summaries for
    tomorrow and the day after tomorrow."""
    try:
        # Current conditions
        obs = mgr.weather_at_place(location)
        current = obs.weather.to_dict()
        current["location"] = location

        # 5-day / 3-hour forecast grid
        forecaster = mgr.forecast_at_place(location, '3h')

        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        tomorrow = (now_utc + timedelta(days=1)).date()
        day_after = (now_utc + timedelta(days=2)).date()

        today_slots: list[dict] = []
        tomorrow_slots: list[dict] = []
        day_after_slots: list[dict] = []

        for w in forecaster.forecast.weathers:
            ref_dt = w.reference_time('date')
            ref_date = ref_dt.date()
            slot = _format_forecast_slot(w)

            if ref_date == today and ref_dt >= now_utc:
                today_slots.append(slot)
            elif ref_date == tomorrow:
                tomorrow_slots.append(slot)
            elif ref_date == day_after:
                day_after_slots.append(slot)

        return {
            "location": location,
            "current": current,
            "today_hourly": today_slots,
            "tomorrow": _aggregate_daily(tomorrow_slots),
            "day_after_tomorrow": _aggregate_daily(day_after_slots),
        }
    except Exception as e:
        return {"error": f"Failed to get weather forecast for {location}: {str(e)}"}
