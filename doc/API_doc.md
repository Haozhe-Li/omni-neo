## 流式信息返回



前端请求post 后端的 /chat接口，然后只有一个参数

````json
{
  "query": "xxx"
}
````



query是用户的提问

后端会以SSE的方式给前端流式返回信息

例子：

````
data: {"type": "answer", "agent": "Supervisor", "content": [{"id": "rs_05165928726b15d2006994c05703a08192a2ad295997461e95", "summary": [], "type": "reasoning"}, {"type": "text", "text": "{\n  \"final_answer\": \"# 你好！\\n\\n你好！很高兴见到你。😊 我可以帮你做很多事情——请告诉我你现在想要什么，我马上开始。\\n\\n## 我能帮你的事情（示例）\\n- 回答事实性问题或做网络/文献调研\\n- 撰写或润色中文/英文文章、邮件、报告\\n- 翻译与校对（中→英 / 英→中）\\n- 帮你写或调试代码（Python、JavaScript 等）并解释实现原理\\n- 帮你学习：计划、讲解概念、做练习题、批改作文\\n- 制定计划、列清单、准备演讲或会议议程\\n- 数据分析与简单计算（需要可执行代码时我会提示）\\n\\n## 接下来怎么做\\n1. 直接告诉我你的请求（越具体越好），比如：\\n   - “请帮我写一封申请实习的邮件”\\n   - “解释一下贝叶斯定理，并举个例子”\\n   - “帮我把这段中文翻成英文”\\n2. 或告诉我你想做的类别，我会给出可选的具体任务清单。\\n\\n## 快速提示\\n- 如果你想用中文交流就继续发中文；如果更方便用英文也可以切换。\\n- 若请求涉及文件或代码，附上内容或说明文件格式/语言版本。\\n\\n你现在想做哪一件？\",\n  \"final_sources\": []\n}", "annotations": [], "id": "msg_05165928726b15d2006994c05c0eb88192a130f41d3a73f5c8"}], "raw": {}}

````



## 信息展示

后端流式的信息会是一个大的json，大致格式如下：
````json
{
    "type": "some type",
    "agent": "Agent Name",
    "content": "some content",
    "raw": {}
}
````

具体来讲，有以下这些场景，需要来解析，然后把信息展示到前端。

#### 答案展示

````json
{
    "type": "answer",
    "agent": "Supervisor",
    "content": [
        {
            "id": "rs_01d8f9337a775c04006994bfb4e96081a0b54965e04e1ca3fe",
            "summary": [],
            "type": "reasoning"
        },
        {
            "type": "text",
            "text": "{\n  \"final_answer\": \"some really long answer in markdown",\n  \"final_sources\":[]",
            "annotations": [],
            "id": "msg_01d8f9337a775c04006994bfbd4f7881a0b8b1b715bff5a819"
        }
    ],
    "raw": {}
}
````

最终的答案会在content的text字段下，然后这个是一个json，两个key，final_answer是一个很长的markdown，final_sources是一个list如下
````json
{"title": "Source Title", "url": "https://..."},
{"title": "Source Title 2", "url": "https://..."}
````



final_answer你需要给渲染好展示给用户，final_source要在answer最底下放一个折叠的页面，里面放所有的上述sources。





#### 推理结果

推理结果你也需要展示，格式如下：
````json
{
    "type": "reasoning",
    "agent": "Sub-agent",
    "content": "We need to produce a concise research report with key findings and source list. Must use tavily_search then skim if needed. Let's do a search query.",
    "raw": {}
}
````

你需要展示"content"字段下的内容就可以，不需要展示agent name之类的。展示reasoning的时候只默认显示前10个字就可以，剩下的默认折叠。

展示效果例如：

some reasoning content （前十个字）



#### Tavily Search

Tavily Search是当agent调用网络搜索工具时会显示：
````json
{
    "type": "tool",
    "tool": "tavily_search",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "max_results": 5,
            "query": "a query",
            "topic": "general"
        },
        "id": "fc_7841fa95-a3f0-42af-87db-7e502a11e99f"
    }
}
````



你只需要展示搜索了"raw"字段下的那个”query“就可以了。

展示效果例如：
Searching on Internet on topic: query



#### 网页速读

这个工具是agent用来快速阅读网页的。样式如下：

````json
{
    "type": "tool",
    "tool": "skimming_web_pages",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "purpose": "some purpose",
            "urls": [
                "url1",
              	"url2"
            ]
        },
        "id": "fc_09a41cb6-cd8d-4b20-bdef-6b2d78e37394"
    }
}
````

你需要显示purpose和urls。

展示效果例如：
Gathering information on Topic: purpose.

Url1, url2, url3 xxx



#### 网页精读

这个工具精读一个网页的，如下

````json
{
    "type": "tool",
    "tool": "get_full_text",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
        	"url": "a url"
        },
        "id": "fc_09a41cb6-1102-6b2d78e37394"
    }
}
````

你只需要提取url就可以

展示效果例如：

Intensive reading and researching on url



#### 断言验证

agent会调用这个工具来验证某个claim。

````json
{
    "type": "tool",
    "tool": "verify_claim",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "fact": "some claim"
        },
        "id": "call_oVetXtBb3NV7ReUiDQAVh4hk"
    }
}
````



你需要展示的信息只有fact。效果如下：

Verifying some claim 



#### Todo List

这个工具是agent用来记录研究目标以及进度的：

````json
{
    "type": "tool",
    "tool": "write_todos",
    "agent": "Supervisor",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "todos": [
                {
                    "content": "task1",
                    "status": "completed"
                },
                {
                    "content": "task2",
                    "status": "in_progress"
                },
              	{
                    "content": "task3",
                    "status": "pending"
                }
            ]
        },
        "id": "call_42W2889jFBqKSKhfWQptY7An"
    }
}
````

你需要展示todo里面的信息。

需要注意的是，todo信息你要在网页里面专门右侧留一块位置一直来展示它：
一共有completed, in_progress, pending三个状态，然后你也要搞一个todo list用原点 对勾，三个点，空着来代表状态。todo你会连续收到，你需要根据todo的信息来更新网页的显示区域。



#### 检查python代码 compile

检查一段python代码能否compile

````json
{
    "type": "tool",
    "tool": "run_python_tool",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "code": "print(1+1)"
        },
        "id": "bvstagb3n"
    }
}
````

你只需要展示code，如下：

Checking python code... （里面具体的code会被默认折叠起来，用户展开会是一个代码框）



#### 运行python代码

运行一段python代码

````json
{
    "type": "tool",
    "tool": "run_python_tool",
    "agent": "Sub-agent",
    "content": "Tool Calling",
    "raw": {
        "args": {
            "code": "print(1+1)"
        },
        "id": "bvstagb3n"
    }
}
````

你只需要展示code，如下：

Running python code... （里面具体的code会被默认折叠起来，用户展开会是一个代码框）。
