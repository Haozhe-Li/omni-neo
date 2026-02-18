import json


def format_answer(content: str) -> list[str]:
    # 1. --- Mock 环境 ---
    def Overwrite(**kwargs):
        return kwargs.get("value", [])

    def HumanMessage(**kwargs):
        return {"type": "human", **kwargs}

    def AIMessage(**kwargs):
        return {"type": "ai", **kwargs}

    def ToolMessage(**kwargs):
        return {"type": "tool", **kwargs}

    eval_context = {
        "Overwrite": Overwrite,
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "ToolMessage": ToolMessage,
        "null": None,
        "true": True,
        "false": False,
    }

    # 2. --- 智能分块解析 ---
    raw_blocks = []
    current_block = ""
    balance_counter = 0
    start_char = None

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1]

        if balance_counter == 0 and not current_block:
            if line.startswith("("):
                start_char = "("
            elif line.startswith("{"):
                start_char = "{"
            else:
                continue

        current_block += line

        if start_char == "(":
            balance_counter += line.count("(") - line.count(")")
        elif start_char == "{":
            balance_counter += line.count("{") - line.count("}")

        if balance_counter == 0 and current_block:
            raw_blocks.append(current_block)
            current_block = ""
            start_char = None

    parsed_events = []
    for block in raw_blocks:
        try:
            event = eval(block, {"__builtins__": None}, eval_context)
            parsed_events.append(event)
        except:
            continue

    # 3. --- 提取逻辑 (应用补丁) ---
    json_results = []

    for item in parsed_events:
        # --- A. 身份判定 ---
        agent_name = "Supervisor"
        data_payload = item

        # 判定逻辑：如果是元组且第一个元素(Context)不为空 -> Sub-agent
        if isinstance(item, tuple) and len(item) == 2:
            context, data_payload = item
            if context and len(context) > 0:
                agent_name = "Sub-agent"
            else:
                agent_name = "Supervisor"

        # --- B. 处理 Model 消息 ---
        if "model" in data_payload and data_payload["model"]:
            messages = data_payload["model"].get("messages", [])
            for msg in messages:
                if msg.get("type") == "ai":
                    kwargs = msg.get("additional_kwargs", {})

                    # 1. Reasoning (所有 Agent 都可以有)
                    reasoning = kwargs.get("reasoning_content")
                    if reasoning:
                        json_results.append(
                            json.dumps(
                                {
                                    "type": "reasoning",
                                    "agent": agent_name,
                                    "content": reasoning,
                                    "raw": {},
                                },
                                ensure_ascii=False,
                            )
                        )

                    # 2. Tool Calls (所有 Agent 都可以有)
                    tool_calls = msg.get("tool_calls") or kwargs.get("tool_calls", [])
                    for tc in tool_calls:
                        t_name = tc.get("name") or tc.get("function", {}).get("name")
                        t_args = tc.get("args") or tc.get("function", {}).get(
                            "arguments"
                        )
                        # Skip "task" tool calls
                        if t_name == "task":
                            continue
                        json_results.append(
                            json.dumps(
                                {
                                    "type": "tool",
                                    "tool": t_name,
                                    "agent": agent_name,
                                    "content": f"Tool Calling",
                                    "raw": {"args": t_args, "id": tc.get("id")},
                                },
                                ensure_ascii=False,
                            )
                        )

                    # 3. Answer (【补丁】仅 Supervisor 可用)
                    content = msg.get("content", "")
                    if content:
                        if agent_name == "Supervisor" and "final_answer" in str(
                            content
                        ):
                            json_results.append(
                                json.dumps(
                                    {
                                        "type": "answer",
                                        "agent": agent_name,
                                        "content": content,
                                        "raw": {},
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        else:
                            pass
                            # Sub-agent 的普通文本输出，转为 Reasoning 展示
                            # 这样既符合 "No Answer" 原则，又不会丢掉 Sub-agent 的回复信息
                            # json_results.append(
                            #     json.dumps(
                            #         {
                            #             "type": "reasoning",
                            #             "agent": agent_name,
                            #             "content": f"{content}",
                            #             "raw": {"original_content": content},
                            #         },
                            #         ensure_ascii=False,
                            #     )
                            # )

        # --- C. 处理 Tools 消息 ---
        if "tools" in data_payload and data_payload["tools"]:
            messages = data_payload["tools"].get("messages", [])
            for msg in messages:
                if msg.get("type") == "tool":
                    tool_name = msg.get("name", "unknown_tool")
                    tool_output = msg.get("content", "")

                    display_content = tool_output
                    should_skip = False
                    try:
                        if len(tool_output) > 200:
                            should_skip = True
                    except:
                        pass

                    if should_skip:
                        continue

                    json_results.append(
                        json.dumps(
                            {
                                "type": "tool",
                                "tool": tool_name,
                                "agent": agent_name,
                                "content": display_content,
                                "raw": {"full_output": tool_output},
                            },
                            ensure_ascii=False,
                        )
                    )

    return json_results
