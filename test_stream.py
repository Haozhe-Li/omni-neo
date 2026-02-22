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
                "content": "请用matplotlib给我画个曲线图",
            }
        ]
    },
    subgraphs=True,
    stream_mode="updates",
    config={"configurable": {"thread_id": "111121"}},
):
    print(content)

    answer_produced = False

    # Pass full payload to format_answer (it will natively extract SupervisorOutputs and struct JSONs now)
    formated = format_answer(content)
    if formated:
        items_to_save = []
        if isinstance(formated, list):
            items_to_save = formated
        else:
            items_to_save = [formated]

        for item in items_to_save:
            file_path = os.path.join(output_dir, f"{file_counter}.json")
            with open(file_path, "w", encoding="utf-8") as f:
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
