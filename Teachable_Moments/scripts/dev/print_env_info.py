#!/usr/bin/env python3

import os
import platform
import sys


def main() -> int:
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())

    try:
        import torch

        print("torch:", torch.__version__)
        print("torch cuda build:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gb = props.total_memory / (1024**3)
                print(f"device{i}:", props.name, f"({gb:.1f} GB)")
    except Exception as e:
        print("torch: not available:", repr(e))

    try:
        import transformers

        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers: not available:", repr(e))

    try:
        import peft

        print("peft:", peft.__version__)
    except Exception as e:
        print("peft: not available:", repr(e))

    try:
        import trl

        print("trl:", trl.__version__)
    except Exception as e:
        print("trl: not available:", repr(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
