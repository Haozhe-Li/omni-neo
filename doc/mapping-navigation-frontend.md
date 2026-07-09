# 地图导航（Navigation）对接说明

给前端同学的对接文档。这次改动给 Pro Agent 加了「导航」能力：agent 现在可以调用
`get_navigation` 工具获取两点间的驾车/步行/骑行距离、时间、大致路书，并且
`map` 代码块新增了一种「路线模式」，用来展示 A→B 的导航路径。

涉及两个互相独立、但配合使用的机制：

1. **`type: "navigation"` SSE 事件** —— 距离/时间/路书文字，随 `get_navigation`
   工具调用一起流式推下来，展示成一张"导航卡片"就行。
2. **`map` 代码块的 `"mode": "route"`** —— 出现在最终答案的 Markdown 正文里，
   跟现有的地点推荐地图（`"mode": "pins"`）是同一套 fence 语法，只是数据结构
   不同，用来在地图上画出 A→B 这条线。

这两者不是绑定关系：agent 可能只调用了 `get_navigation`（比如用户只是问"从A到B
怎么走"，答案里未必需要地图），也可能在行程报告里同时给出文字导航卡片 + 路线地图。
前端按各自的场景独立渲染即可。

---

## 1. `map` 代码块：新增 `mode` 字段

现有的地点推荐地图（餐厅/景点/酒店等）不变，只是显式加了 `"mode": "pins"` 标记
（省略时默认按 `pins` 处理，兼容老数据）。新增的路线模式是 `"mode": "route"`。

**这两种 mode 不会出现在同一个 JSON 里** —— 一个 `map` 代码块要么是"一组推荐地
点"，要么是"一条 A→B 路线"，不会两者混合。如果一段内容里既要展示推荐地点又要展示
路线，agent 会拆成两个独立的 ` ```map ` 代码块，前端按顺序各自渲染成两张地图卡片
即可，不需要处理"一张地图里既有点又有线"的情况。

### 1.1 `pins` 模式（地点推荐，原有逻辑不变）

````
```map
{
  "mode": "pins",
  "title": "Ramen shops in Shinjuku",
  "pins": [
    { "name": "Fuunji, Shinjuku, Tokyo", "description": "Famous for tsukemen, expect a line" }
  ]
}
```
````

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `string` | `"pins"`，可省略（省略时按 `pins` 处理） |
| `title` | `string` | 地图标题 |
| `pins` | `array` | 地点数组（最多 8 个），每项 `{ name, description? }` |

`name` 是地名/地址文本，前端负责 geocode 成坐标（跟现在的实现一致，没有变化）。

### 1.2 `route` 模式（新增：导航路线）

````
```map
{
  "mode": "route",
  "title": "Hotel to the Louvre",
  "travel_mode": "driving",
  "origin": {
    "name": "Hôtel de Ville, Paris",
    "description": "Starting point"
  },
  "destination": {
    "name": "Musée du Louvre, Paris",
    "description": "Optional note"
  }
}
```
````

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `string` | 固定为 `"route"` |
| `title` | `string` | 地图标题 |
| `travel_mode` | `string` | `"driving"` \| `"walking"` \| `"cycling"` |
| `origin` | `object` | 起点，`{ name, description? }`，同 pin 的字段结构 |
| `destination` | `object` | 终点，`{ name, description? }` |

**这里只给起点/终点的地名，不给坐标或路线几何数据** —— 跟 `pins` 模式一样，
`name` 交给前端 geocode。画实际路线（沿路网的那条线，而不是两点直线）需要前端
自己调用一个路线规划 API（比如浏览器端直接调 OSRM 的 `route` 接口，或者你们已有
的地图 SDK 自带的 directions 功能），传入 `origin`/`destination` geocode 出来
的坐标 + `travel_mode` 去拿路线并画出来。后端这边不提供路线的 geometry。

> 为什么后端不直接把路线坐标塞进这个 JSON？ 因为地图渲染跟坐标解析目前全在前端
> 一侧（跟 pins 模式保持一致的架构），后端只负责告诉你"要画什么"（起点、终点、
> 出行方式），具体怎么画、用哪个路线 API、样式如何，都是前端自己的事。

---

## 2. SSE 新事件：`type: "navigation"`

跟 `weather` / `stock` / `currency` 事件是同一套机制：只要后台调用了
`get_navigation` 工具，就会额外推一条 `type: "navigation"` 事件，携带精简后的
距离/时间/路书文字，不需要自己解析 `tool` 事件里的原始 JSON。

### 2.1 `tool` 事件（工具调用中，可选展示 "正在获取导航..." 之类的 loading 态）

```json
{
  "type": "tool",
  "tool": "get_navigation",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": {
      "origin_lat": 37.7749,
      "origin_lng": -122.4194,
      "destination_lat": 37.8199,
      "destination_lng": -122.4783,
      "mode": "driving"
    },
    "id": "call_..."
  }
}
```

### 2.2 `navigation` 事件（结果，用来渲染导航卡片）

```json
{
  "type": "navigation",
  "agent": "Sub-agent",
  "navigation": {
    "mode": "driving",
    "distance_km": 5.2,
    "duration_min": 14,
    "route_summary": [
      "Head north on Main St (300 m)",
      "Turn right onto Oak Ave (1.2 km)",
      "Arrive at destination"
    ]
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `string` | `driving` / `walking` / `cycling` |
| `distance_km` | `number` | 总距离（公里，保留 1 位小数） |
| `duration_min` | `number \| null` | 预计耗时（分钟，取整）；上游没返回时可能是 `null` |
| `route_summary` | `string[]` | 精简路书，几行到十几行文字描述，**不是**逐个路口的详细转弯指令，仅供参考展示，不建议逐字当成"导航播报"用 |

### 2.3 失败态

导航服务查不到路线，或者请求失败时：

```json
{
  "type": "navigation",
  "agent": "Sub-agent",
  "navigation": { "error": "No route found." }
}
```

此时没有 `distance_km`/`duration_min`/`route_summary`，前端展示一个"暂时无法
获取导航信息"之类的兜底文案即可，不要当成正常数据渲染。

---

## 3. 小结 / 前端 TODO

- [ ] `map` 代码块解析逻辑加一个 `mode` 分支：`"route"` 时读 `travel_mode` +
      `origin` + `destination`，而不是 `pins` 数组。
- [ ] `route` 模式下，geocode 起终点坐标后自行调用路线规划 API 画出路径（后端
      不提供路线 geometry）。
- [ ] SSE 流里新增处理 `type: "navigation"`，渲染成一张距离/时间/路书卡片；
      `navigation.error` 存在时走失败态展示。
- [ ] confirm：一个 `map` 代码块永远只有一种 `mode`，不会同时出现 `pins` 和
      `route` 相关字段——不需要兼容"混合"情况。

如果需要现成的地图/路线渲染库选型建议，或者想让 OSRM 的调用改成走后端代理（比如
避免暴露某个商用路线 API 的 key），告诉我一声，我再补充。
