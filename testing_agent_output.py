import dotenv

dotenv.load_dotenv()

from core.supervisor import agent
from core.utils.format import format_answer

import pprint

import os
import json

output_dir = "output_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_counter = 1

for content in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "深度研究介绍一下Langchain",
            }
        ]
    },
    subgraphs=True,
    stream_mode="updates",
    config={"configurable": {"thread_id": "1221"}},
):
    print(content)
    formated = format_answer(str(content))
    if formated:
        items_to_save = []
        if isinstance(formated, list):
            items_to_save = formated
        else:
            items_to_save = [formated]

        for item in items_to_save:
            file_path = os.path.join(output_dir, f"{file_counter}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                # If item is a string (JSON string), load it to ensure it's formatted nicely, or write as is.
                # User said "formated之后的content会是一个dict", implying `item` might be a dict.
                # But format_answer returns JSON strings usually. We handle both.
                if isinstance(item, str):
                    try:
                        json_data = json.loads(item)
                        json.dump(json_data, f, ensure_ascii=False, indent=4)
                    except:
                        f.write(item)
                else:
                    json.dump(item, f, ensure_ascii=False, indent=4)

            print(f"Saved {file_path}")
            pprint.pprint(item)
            file_counter += 1
