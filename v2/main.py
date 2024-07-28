import argparse
from brains import Brain, tool


def args(**kwargs):
    kwargs["prog"] = kwargs.get("prog", "otui-v2")
    kwargs["description"] = kwargs.get("description", "Ollama Terminal User Interface")

    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument(
        "--auto_hijack",
        "--ah",
        action="store_true",
        default=False,
        help="Initializes otui in auto-hijack mode.",
    )

    # argument for ollama model
    parser.add_argument(
        "--model",
        "--m",
        action="store",
        default="llama3.1",
        help="The model to use for OTUI. Defaults to llama3.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args()
