---
name: mapping
description: show places on an interactive map, or a single A→B route, inline in the answer. Use whenever the user asks to find, recommend, or explore physical locations (restaurants, hotels, attractions, shops, etc.), or asks how to get from one place to another (driving/walking/cycling directions).
---

# Mapping

Show locations on a map by embedding a fenced code block whose language is `map`, containing a JSON object. It renders inline, right where you place it in the answer.

There are two mutually-exclusive modes: `"pins"` (place recommendations, the default) and `"route"` (a single A→B navigation path). **Never combine them in one block** — see [Hard rules](#hard-rules).

The frontend resolves place names to coordinates automatically — you never need to look up or supply lat/lng, in either mode.

## Pins mode (default) — recommending places

````map
{
  "mode": "pins",
  "title": "SHORT DESCRIPTIVE TITLE",
  "pins": [
    {
      "name": "Place Name, City",
      "description": "One-line note (hours, rating, why it's recommended, etc.)"
    }
  ]
}
````

- `mode` — `"pins"`; may be omitted (pins is the default when there's no `mode` field, for backward compatibility).
- `title` — concise, describes what the map shows (e.g. "Ramen shops in Shinjuku")
- `pins` — array of locations (up to 8)
- `name` — include enough context to be unambiguous: `"Nobu Restaurant, Los Angeles"` not just `"Nobu"`
- `description` — optional but recommended

### Where pins must come from

Every pin must be one of:

1. **A result from `google_search_places`** — restaurants, hotels, attractions, shops, or any specific venue
2. **A result from web search** (`google_search` / `tavily_search`) — e.g. a hotel name mentioned in an article
3. **A high-scope geographic entity** — cities, neighbourhoods, states, countries, or widely known landmarks (e.g. `"Eiffel Tower, Paris"`, `"Manhattan, New York"`)

**Never invent a specific business or venue name.** If you want to put a restaurant or hotel on the map and haven't searched for it yet, call `google_search_places` first.

## Route mode — driving/walking/cycling directions between two points

Use this when the answer involves getting from one specific point to another
by car, on foot, or by bike. Call the `get_navigation` tool first to get the
real distance/duration for your prose — this map block is just the visual, it
doesn't replace the tool call.

````map
{
  "mode": "route",
  "title": "SHORT DESCRIPTIVE TITLE",
  "travel_mode": "driving",
  "origin": {
    "name": "Place Name, City",
    "description": "Optional one-line note"
  },
  "destination": {
    "name": "Place Name, City",
    "description": "Optional one-line note"
  }
}
````

- `mode` — must be `"route"`.
- `travel_mode` — one of `"driving"`, `"walking"`, `"cycling"` — must match the
  `mode` you passed to `get_navigation`.
- `origin` / `destination` — exactly one of each, same `name`/`description`
  shape as a pin. Same sourcing rule as pins mode (real venue names only).

## Hard rules

- **A map block is either `pins` or `route`, never both.** Don't add a `pins`
  array to a `route` block or vice versa. If you need to show recommended
  venues *and* a route in the same section, emit two separate `map` blocks.
- **Valid JSON only**: double-quoted keys, no trailing commas.
- Place the map block where it naturally fits in the answer.
- Follow up with a brief prose summary so the answer works even if the map fails to render.
