import argparse
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import get_settings
from inference.client import InferenceClient, OpenAICompatClient, encode_image
from storage.local import IMAGE_EXTENSIONS, LocalStorage

PROMPT = (
    "Describe this photo in one sentence, then list its mood and setting. "
    "Be concrete and factual. Do not speculate about people's identities."
)


@dataclass(frozen=True)
class BakeoffRow:
    model: str
    photo: str
    seconds: float
    caption: str


def caption_once(
    client: InferenceClient,
    model: str,
    image_bytes: bytes,
    clock: Callable[[], float],
) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_bytes)}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    started = clock()
    caption = client.complete(model, messages)
    return caption, clock() - started


def run_bakeoff(
    client: InferenceClient,
    models: list[str],
    images: list[tuple[str, bytes]],
    clock: Callable[[], float] = time.monotonic,
) -> list[BakeoffRow]:
    rows: list[BakeoffRow] = []
    for model in models:
        for name, data in images:
            caption, seconds = caption_once(client, model, data, clock)
            rows.append(BakeoffRow(model=model, photo=name, seconds=seconds, caption=caption))
    return rows


def format_table(rows: list[BakeoffRow]) -> str:
    models = sorted({row.model for row in rows})
    lines = ["model                     photos   mean s/photo", "-" * 48]
    for model in models:
        subset = [row for row in rows if row.model == model]
        mean = sum(row.seconds for row in subset) / len(subset)
        lines.append(f"{model:<25} {len(subset):>6}   {mean:>12.2f}")
    lines.append("")
    for model in models:
        lines.append(f"--- {model} ---")
        for row in [r for r in rows if r.model == model][:5]:
            lines.append(f"  {row.photo}: {row.caption}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare caption models on real photos.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    storage = LocalStorage(args.library, extensions=IMAGE_EXTENSIONS)
    keys = list(storage.iter_keys())
    random.Random(args.seed).shuffle(keys)
    images = [(key, storage.read(key)) for key in keys[: args.count]]
    if not images:
        raise SystemExit(f"no images found under {args.library}")

    client = OpenAICompatClient(settings.inference_base_url)
    print(format_table(run_bakeoff(client, args.models, images)))


if __name__ == "__main__":
    main()
