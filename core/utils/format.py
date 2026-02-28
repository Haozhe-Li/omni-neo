import json
from typing import Any


def _attach_sources(payload: dict) -> dict:
    merged = dict(payload)
    existing = merged.get("sources")
    existing_list = existing if isinstance(existing, list) else []
    merged["sources"] = existing_list
    return merged


def _attach_assets(payload: dict) -> dict:
    merged = dict(payload)
    existing = merged.get("assets")
    existing_list = existing if isinstance(existing, list) else []
    merged["assets"] = existing_list
    return merged


def extract_struct_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            res = extract_struct_dict(obj.model_dump())
            if res is not None:
                return res
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            res = extract_struct_dict(obj.dict())
            if res is not None:
                return res
        except Exception:
            pass

    if isinstance(obj, dict):
        if "answer" in obj or "answer" in obj:
            return obj
        if "structured_response" in obj:
            res = extract_struct_dict(obj["structured_response"])
            if res is not None:
                return res

        for k, v in obj.items():
            res = extract_struct_dict(v)
            if res is not None:
                return res

    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            res = extract_struct_dict(item)
            if res is not None:
                return res

    if hasattr(obj, "additional_kwargs"):
        parsed = obj.additional_kwargs.get("parsed")
        if parsed:
            res = extract_struct_dict(parsed)
            if res is not None:
                return res

    if hasattr(obj, "answer") or hasattr(obj, "answer"):
        data = {
            "title": getattr(obj, "title", None),
            "answer": getattr(obj, "answer", None) or getattr(obj, "answer", None),
            "assets": getattr(obj, "assets", getattr(obj, "assets", [])),
        }
        if hasattr(obj, "sources"):
            data["sources"] = getattr(obj, "sources", [])
        return data

    return None


def format_answer(content: Any) -> list[str]:
    if not isinstance(content, str):
        struct_dict = extract_struct_dict(content)
        if struct_dict and ("answer" in struct_dict or "answer" in struct_dict):
            struct_dict = _attach_sources(struct_dict)
            struct_dict = _attach_assets(struct_dict)
            item = json.dumps(
                {
                    "type": "answer",
                    "agent": "Supervisor",
                    "content": json.dumps(struct_dict, ensure_ascii=False),
                    "raw": {},
                },
                ensure_ascii=False,
            )
            return [item]
        content_str = str(content)
    else:
        content_str = content

    # 1. --- Mock 环境 ---
    def Overwrite(**kwargs):
        return kwargs.get("value", [])

    def HumanMessage(**kwargs):
        return {"type": "human", **kwargs}

    def AIMessage(**kwargs):
        return {
            "type": "ai",
            "additional_kwargs": kwargs.get("additional_kwargs", {}),
            **kwargs,
        }

    def ToolMessage(**kwargs):
        return {"type": "tool", **kwargs}

    def SupervisorOutput(**kwargs):
        return kwargs

    eval_context = {
        "Overwrite": Overwrite,
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "ToolMessage": ToolMessage,
        "SupervisorOutput": SupervisorOutput,
        "null": None,
        "true": True,
        "false": False,
    }

    # 2. --- 智能分块解析 ---
    raw_blocks = []
    current_block = ""
    balance_counter = 0
    start_char = None

    for line in content_str.split("\n"):
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

    # [补丁] 针对直接传入的字典字符串 (非嵌套 tuple) 做尝试解析
    try:
        parsed_dict = json.loads(content_str.replace("'", '"'))
        if isinstance(parsed_dict, dict) and "structured_response" in parsed_dict:
            raw_blocks.append(content_str)
    except:
        pass

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

                        # [补丁] 针对 SupervisorOutput 结构化输出工具，拦截并转为 answer
                        if t_name == "SupervisorOutput" and agent_name == "Supervisor":
                            if isinstance(t_args, str):
                                try:
                                    t_args = json.loads(t_args)
                                except:
                                    pass

                            # Append to answer stream if it looks like the expected model
                            if isinstance(t_args, dict) and "answer" in t_args:
                                t_args = _attach_sources(t_args)
                                t_args = _attach_assets(t_args)
                                json_results.append(
                                    json.dumps(
                                        {
                                            "type": "answer",
                                            "agent": agent_name,
                                            "content": json.dumps(
                                                t_args, ensure_ascii=False
                                            ),
                                            "raw": {},
                                        },
                                        ensure_ascii=False,
                                    )
                                )
                            continue  # Do not emit "SupervisorOutput" as a standard tool call to the UI

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
                    # 优先从 LangChain ProviderStrategy 的 parsed kwarg 中提取
                    parsed_data = kwargs.get("parsed") or msg.get("parsed")
                    if (
                        parsed_data
                        and isinstance(parsed_data, dict)
                        and "answer" in parsed_data
                    ):
                        if agent_name == "Supervisor":
                            parsed_data = _attach_sources(parsed_data)
                            parsed_data = _attach_assets(parsed_data)
                            json_results.append(
                                json.dumps(
                                    {
                                        "type": "answer",
                                        "agent": agent_name,
                                        "content": json.dumps(
                                            parsed_data, ensure_ascii=False
                                        ),
                                        "raw": {},
                                    },
                                    ensure_ascii=False,
                                )
                            )
                    else:
                        # 退路：如果有传统的纯文本 content，依然返回
                        content = msg.get("content", "")
                        if content and isinstance(content, str):
                            if agent_name == "Supervisor" and "answer" in content:
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

        # --- B.2. 处理 Structured Response 直接返回 (如 langgraph 返回结构体) ---
        if isinstance(data_payload, dict) and "structured_response" in data_payload:
            structured = data_payload["structured_response"]
            struct_dict = extract_struct_dict(structured)
            if struct_dict and "answer" in struct_dict and agent_name == "Supervisor":
                struct_dict = _attach_sources(struct_dict)
                struct_dict = _attach_assets(struct_dict)
                json_results.append(
                    json.dumps(
                        {
                            "type": "answer",
                            "agent": agent_name,
                            "content": json.dumps(struct_dict, ensure_ascii=False),
                            "raw": {},
                        },
                        ensure_ascii=False,
                    )
                )

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
