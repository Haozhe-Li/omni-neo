---
name: charting
description: How to draw interactive charts inline with ECharts. Use when a chart communicates the data better than prose — trends over time, comparisons, distributions, proportions. Skip it for just one or two numbers.
---

# Charting

Draw a chart by embedding a fenced code block whose language is `echarts`,
containing a COMPLETE Apache ECharts `option` object as JSON. It renders inline,
right where you place it — in a chat reply OR inside a report.

```echarts
{"title":{"text":"Fruit sales"},
 "tooltip":{},
 "xAxis":{"type":"category","data":["Apples","Bananas","Cherries"]},
 "yAxis":{"type":"value"},
 "series":[{"type":"bar","data":[30,50,20]}]}
```

## Rules
- Valid JSON only: double-quoted keys, no trailing commas, no JavaScript or functions.
- It MUST include `series`. Add `title`, `tooltip`, and axes where they help.
- Reach for a chart only when it genuinely clarifies the data — not for decoration.
