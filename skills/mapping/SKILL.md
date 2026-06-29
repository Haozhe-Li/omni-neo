---
name: mapping
description: show places on an interactive map inline in the answer. Use whenever the user asks to find, recommend, or explore physical locations (restaurants, hotels, attractions, shops, etc.).
---

# Mapping

Show locations on a map by embedding a fenced code block whose language is `map`, containing a JSON object. It renders inline, right where you place it in the answer.

The frontend resolves place names to coordinates automatically — you never need to look up or supply lat/lng.

## Schema

````map
{
  "title": "SHORT DESCRIPTIVE TITLE",
  "pins": [
    {
      "name": "Place Name, City",
      "description": "One-line note (hours, rating, why it's recommended, etc.)"
    }
  ]
}
````

- `title` — concise, describes what the map shows (e.g. "Ramen shops in Shinjuku")
- `pins` — array of locations (up to 8)
- `name` — include enough context to be unambiguous: `"Nobu Restaurant, Los Angeles"` not just `"Nobu"`
- `description` — optional but recommended

## Where pins must come from

Every pin must be one of:

1. **A result from `google_search_places`** — restaurants, hotels, attractions, shops, or any specific venue
2. **A result from web search** (`google_search` / `tavily_search`) — e.g. a hotel name mentioned in an article
3. **A high-scope geographic entity** — cities, neighbourhoods, states, countries, or widely known landmarks (e.g. `"Eiffel Tower, Paris"`, `"Manhattan, New York"`)

**Never invent a specific business or venue name.** If you want to put a restaurant or hotel on the map and haven't searched for it yet, call `google_search_places` first.

## Hard rules

- **Valid JSON only**: double-quoted keys, no trailing commas.
- Place the map block where it naturally fits in the answer.
- Follow up with a brief prose summary so the answer works even if the map fails to render.
