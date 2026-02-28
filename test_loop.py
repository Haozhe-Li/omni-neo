import dotenv

dotenv.load_dotenv()

from core.supervisor import agent
import json

for content in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请研究一下当前deep learning的最新进展",
            }
        ]
    },
    subgraphs=True,
    stream_mode="updates",
    config={"configurable": {"thread_id": "test_looping_123"}},
):
    print("--------------------------------")
    print(content)
