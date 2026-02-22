import io
import uuid
import boto3
from botocore.config import Config
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core.utils.shorter_url import get_shorten_url
import pandas as pd
from core.tools.coding_sandbox import is_safe_code

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

CUSTOM_STYLE = {
    "axes.facecolor": "#F3F3EE",
    "figure.facecolor": "#F3F3EE",
    "text.color": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "lines.linewidth": 2.5,
    "lines.solid_capstyle": "round",
    "axes.prop_cycle": plt.cycler(
        color=["#20B2AA", "#005A5A", "#E69F00", "#56B4E9", "#009E73"]
    ),
    "grid.color": "#000000",
    "grid.alpha": 0.05,
    "grid.linestyle": "-",
    "grid.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "sans-serif",
    ],
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#D9D9D9",
}


def draw_graph(code: str) -> str:
    """Draw a graph using pandas, numpy and matplotlib.

    Args:
        code (str): The code to draw the graph.

    Returns:
        str: The URL of the graph or error message.
    """
    # print(code)

    # is_safe, reason = is_safe_code(code)
    # if not is_safe:
    #     return f"Error: Code rejected for safety reasons: {reason}. Do not use plt.savefig() or any file save/read operations. The tool processes saving automatically."

    plt.clf()
    plt.close("all")
    plt.rcParams.update(CUSTOM_STYLE)
    execution_namespace = {"plt": plt, "np": np, "pd": pd}

    try:
        exec(code, execution_namespace)

        # if not plt.get_fignums():
        #     return "Error: 没有生成任何图表，请检查你的代码。"

        fig = plt.gcf()
        fig.autofmt_xdate(rotation=45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=False, dpi=150)
        buf.seek(0)

        bucket_name = "omni"
        object_key = f"agent-charts/{uuid.uuid4().hex}.png"

        s3_client.put_object(
            Bucket=bucket_name, Key=object_key, Body=buf, ContentType="image/png"
        )

        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=3600 * 24 * 7,  # 7 days
        )

        shorter_url = get_shorten_url(presigned_url)
        print(shorter_url)
        return shorter_url if shorter_url else presigned_url

    except Exception as e:
        print(e)
        return f"Code Execution Error: {str(e)}"

    finally:
        plt.clf()
        plt.close("all")


if __name__ == "__main__":
    code = """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 1. Create a DataFrame
    data = {
        "Category": ["A", "B", "C", "D", "E"],
        "Values": [23, 45, 56, 78, 34],
    }
    df = pd.DataFrame(data)

    # 2. Create the plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["Category"], df["Values"], color="#20B2AA")

    # 3. Add labels and title
    plt.xlabel("Category")
    plt.ylabel("Values")
    plt.title("Bar Chart Example")

    # 4. Add value labels on top of bars
    for i, value in enumerate(df["Values"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")

    # 5. Display the plot (The tool will handle saving)
    plt.show()
    """

    url = draw_graph(code)
    print(f"Generated Graph URL: {url}")
