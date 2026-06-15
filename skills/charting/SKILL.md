---
name: charting
description: draw interactive charts inline with ECharts. Always use this when there's comparison, trends, or distributions or anytime a chart would be more effective than prose.
---

# Charting

Draw a chart by embedding a fenced code block whose language is `echarts`,
containing a **complete** Apache ECharts `option` object as valid JSON.
It renders inline, right where you place it — in a chat reply OR inside a report.

---

## Style system — apply to every chart

The UI follows a restrained Scandinavian-neutral design language. Every chart
must feel like it belongs in the same page: quiet, fact-driven, no decoration
for its own sake.

### Color palette

Use this exact series color order. Never use saturated "tech blue".

```
Primary series   #20B2AA   (washed teal — matches the app accent)
2nd series       #005A5A   (deep teal)
3rd series       #7B9E9E   (muted teal-gray)
4th series       #C4A882   (warm tan)
5th series       #8B7D6B   (warm brown)
6th series       #5B8FA8   (steel blue)
```

If only one series, always use `#20B2AA`. For two series, use `#20B2AA` +
`#005A5A`.

### Base template

Copy this shell and fill in your data. Do **not** change any style key unless a
specific chart type requires it.

```echarts
{
  "backgroundColor": "transparent",
  "color": ["#20B2AA", "#005A5A", "#7B9E9E", "#C4A882", "#8B7D6B", "#5B8FA8"],
  "textStyle": { "fontFamily": "Inter, system-ui, sans-serif", "color": "#1A1A1A" },
  "title": {
    "text": "YOUR TITLE",
    "textStyle": { "fontFamily": "Inter, system-ui, sans-serif", "fontSize": 14, "fontWeight": "600", "color": "#1A1A1A" },
    "subtextStyle": { "color": "#8A8A8A", "fontSize": 12 },
    "left": 0,
    "top": 0
  },
  "tooltip": {
    "trigger": "axis",
    "backgroundColor": "#FFFFFF",
    "borderColor": "rgba(0,0,0,0.06)",
    "borderWidth": 1,
    "textStyle": { "color": "#1A1A1A", "fontSize": 13 },
    "extraCssText": "box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-radius: 6px;"
  },
  "legend": {
    "top": 0,
    "right": 0,
    "icon": "circle",
    "itemWidth": 8,
    "itemHeight": 8,
    "itemGap": 16,
    "textStyle": { "color": "#8A8A8A", "fontSize": 12 }
  },
  "grid": { "left": 0, "right": 16, "top": 48, "bottom": 0, "containLabel": true },
  "xAxis": {
    "type": "category",
    "data": ["YOUR", "CATEGORIES"],
    "axisLine": { "lineStyle": { "color": "rgba(0,0,0,0.08)" } },
    "axisTick": { "show": false },
    "axisLabel": { "color": "#8A8A8A", "fontSize": 12 }
  },
  "yAxis": {
    "type": "value",
    "splitLine": { "lineStyle": { "color": "rgba(0,0,0,0.06)", "type": "solid" } },
    "axisLine": { "show": false },
    "axisTick": { "show": false },
    "axisLabel": { "color": "#8A8A8A", "fontSize": 12 }
  },
  "series": [
    { "type": "bar", "name": "SERIES NAME", "data": [0, 0, 0], "barMaxWidth": 48, "itemStyle": { "borderRadius": [4, 4, 0, 0] } }
  ]
}
```

---

## Per-type guidance

### Bar chart
- `barMaxWidth: 48` — prevents bars from becoming grotesque on wide layouts.
- `itemStyle.borderRadius: [4, 4, 0, 0]` — rounded top corners only.
- Horizontal bars (`"yAxis": {"type":"category"}, "xAxis": {"type":"value"}`)
  suit ranked lists; use `borderRadius: [0, 4, 4, 0]` in that case.

### Line chart
- `"smooth": true` — always. Hard angles feel dated.
- Add `"areaStyle": { "opacity": 0.08 }` for a single series to give it a
  subtle fill that reads as "context" without visual noise.
- Set `"symbol": "none"` to remove per-point dots unless there are ≤ 8 points.

### Pie / donut chart
- Always use a donut: `"radius": ["48%", "70%"]`.
- `"label": { "formatter": "{b}\n{d}%" }` — name + percentage, two lines.
- `"padAngle": 2` for a small gap between slices.
- No legend if ≤ 4 slices — the labels are enough.

### Scatter chart
- `"symbolSize": 8` default.
- Use `"tooltip": { "trigger": "item" }` instead of axis.

---

## Hard rules

- **Valid JSON only**: double-quoted keys, no trailing commas, no JS/functions.
- Every chart **must** include `series`. Add `title`, `tooltip`, axes where
  they help.
- `backgroundColor` must always be `"transparent"`.
- Never override `fontFamily` — always inherit from the base template.
- Reach for a chart only when it genuinely clarifies the data — not decoration.
