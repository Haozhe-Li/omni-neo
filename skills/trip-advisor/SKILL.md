---
name: trip-advisor
description: plan a complete trip for the user — itinerary, flights, hotels, restaurants, weather, maps, and a day-by-day report. Use when the user asks to plan a trip, vacation, travel itinerary, or anything involving travel to a destination.
---

# Trip Advisor

A structured travel-planning workflow: clarify → plan → research → report.

---

## Step 0 — Clarify (mandatory, always run this first)

**Immediately load the `ask question` skill** and emit a `<question>` block.
Do this even if the user's message already contains some details — always
confirm all required fields before proceeding.

Gather in a single question block:

1. **Departure city** and **travel dates** (outbound + return)
2. **Destination(s)** — one city or a multi-stop route
3. **Number of travelers** and group type (solo / couple / family with children ages)
4. **Total budget** (approximate, in their currency)
5. **Travel style** — budget backpacker / mid-range / luxury
6. **Interests** — food, nature, culture, nightlife, shopping, etc.
7. **Special constraints** — dietary restrictions, mobility needs, must-see items

Do not proceed to Step 1 until the user has answered.

---

## Step 1 — Plan

Structure the plan to cover:

1. **Weather** — forecast or historical climate for travel dates
2. **Flights** — options, price range, travel time from origin
3. **Accommodation** — hotels or stays per destination city
4. **Attractions & activities** — per city, matching stated interests
5. **Restaurants** — per city, matching budget and dietary needs
6. *(Repeat 3–5 for each additional destination city in a multi-stop trip)*
7. **Report** — last; explicitly note: load `report-writing` and `mapping` skills

Feel free to use `write_todos` to lay this out and track progress — use your judgment on when it's actually helpful.

---

## Step 2 — Research

Work through the plan in order. Roughly 5 tool calls max per item (e.g. 2 searches + a couple of follow-ups) — if that's not turning up results, move on rather than linger.

### Weather
- Travel dates **within 5 days**: call `get_weather_forecast`.
- Travel dates **further out**: use `google_search` — query `"average weather in [city] in [month]"`.

### Flights
- Use `tavily_search` or `google_search` — query `"flights from [origin] to [destination] [month year] price"`.
- Note price range, major airlines, and typical flight duration.

### Accommodation (per city)
- Use `google_search_places` — query `"hotels in [city]"` filtered to budget tier.
- Supplement with `google_search` for specific property reviews if needed.

### Attractions & activities (per city)
- Use `google_search_places` — query `"top attractions in [city]"` or `"things to do in [city]"`.
- Use `google_search` to check opening hours or entrance fees for key sites.

### Restaurants (per city)
- Use `google_search_places` — query `"best restaurants in [city]"` or `"[cuisine] restaurants in [city]"`.
- Match results to the user's dietary constraints and budget tier.

---

## Step 3 — Report (final step)

**Immediately load the `report-writing` and `mapping` skills.**

Write a `<report>` following the report-writing skill rules. Target **~1500 words**.

### Required report structure

```
## Overview
[2–3 sentence trip summary. Then the FIRST map: city/route overview.]

## Day-by-Day Itinerary
### Day 1 — [City]
[Morning / Afternoon / Evening breakdown. Specific venues from research.]
[Embed a map here showing the day's locations — restaurants, attractions, hotel.]

### Day 2 — [City]
...
[Embed at least one more map for a second day or second city.]

## Getting There
[Flight options, price range, journey time.]

## Where to Stay
[Hotel recommendations per city with brief notes.]

## Weather & What to Pack
[Forecast or seasonal averages. Packing tips.]

## Budget Summary
[Table: flights / accommodation / food / activities / total estimate.]
```

### Mapping rules within the report

- **First map** (in Overview): high-scope route overview. Pins are cities or
  neighbourhoods only (e.g. `"San Francisco, CA"`, `"Los Angeles, CA"`). No
  `google_search_places` required for this map.
- **Subsequent maps** (at least 2 more, in Day sections): specific venues —
  restaurants, hotels, attractions. All pins must come from your research
  (google_search_places or web search results). Never invent venue names.
- Place each map directly after the prose that introduces those locations.

### After the report

Write a short chat reply (2–4 sentences) highlighting the most important
tip or the best part of the itinerary. Then stop — no further tool calls.

---

## Budget

- **Sources**: 10–20 total across the whole trip. More is expected than deep-research
  given the number of research domains.
- **Hard stop**: if approaching the tool-call limit, skip remaining research
  items and write the report with what you have — an honest partial plan beats
  running out of steps mid-research.

---

## Rules

- Never fabricate hotel names, restaurant names, flight prices, or venue
  details. If an item yields no results, note the gap in the report honestly.
- Calibrate all recommendations to the user's stated budget tier and interests.
- Day-by-day pacing should be realistic — do not over-schedule.
- If the trip spans multiple cities, ensure the itinerary flow makes geographic
  sense (don't route the user backwards).
