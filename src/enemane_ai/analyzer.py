from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Protocol, Sequence, cast

import pypdfium2 as pdfium
from google import genai
from google.genai import types as genai_types
from PIL import Image

PRESET_PROMPT = (
    "あなたはグラフリテラシーに長けたアナリストです。"
    " 以下のフォーマットで簡潔にコメントを返してください:"
    " 1) トレンド 2) 読み取れる含意 3) 注意点。"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
CSV_EXTENSIONS = {".csv"}
DEFAULT_MODEL_NAME = "gemini-3-pro-preview"


@dataclass
class GraphEntry:
    display_label: str
    image: Image.Image | None = None
    text: str | None = None


def collect_graph_entries(paths: Iterable[Path]) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(path) as img:
                entries.append(GraphEntry(display_label=path.name, image=img.convert("RGB")))
            continue

        if suffix == ".pdf":
            for page_index, page_image in pdf_to_images(path):
                label = f"{path.name}#{page_index + 1}"
                entries.append(GraphEntry(display_label=label, image=page_image))
            continue

        if suffix in CSV_EXTENSIONS:
            entries.append(csv_to_graph_entry(path))

    return entries


def pdf_to_images(path: Path) -> list[tuple[int, Image.Image]]:
    document = pdfium.PdfDocument(path)
    images: list[tuple[int, Image.Image]] = []
    for page_index in range(len(document)):
        page = document.get_page(page_index)
        try:
            pil_image = page.render(scale=2).to_pil().convert("RGB")
            images.append((page_index, pil_image))
        finally:
            page.close()
    document.close()
    return images


class GraphLanguageModel(Protocol):
    def comment_on_graph(self, image: Image.Image, prompt: str) -> str: ...

    def comment_on_text(self, text: str, prompt: str) -> str: ...


class GeminiGraphLanguageModel:
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL_NAME):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @classmethod
    def from_env(cls) -> "GeminiGraphLanguageModel":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY is not set"
            raise RuntimeError(msg)
        return cls(api_key=api_key)

    def comment_on_graph(self, image: Image.Image, prompt: str) -> str:
        image_part = self._image_part(image)
        contents = cast(genai_types.ContentListUnionDict, [prompt, image_part])
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text or ""

    def comment_on_text(self, text: str, prompt: str) -> str:
        contents = cast(genai_types.ContentListUnionDict, [prompt, text])
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text or ""

    @staticmethod
    def _image_part(image: Image.Image) -> genai_types.Part:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return genai_types.Part.from_bytes(
            mime_type="image/png",
            data=buffer.getvalue(),
        )


def csv_to_graph_entry(path: Path) -> GraphEntry:
    series = parse_temperature_csv(path)
    text = format_temperature_series(series)
    return GraphEntry(display_label=path.name, text=text)


def parse_temperature_csv(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if len(row) < 2:
                continue
            label = row[0].strip()
            try:
                temperature = float(row[1])
            except ValueError:
                continue
            if not label:
                continue
            rows.append((label, temperature))

    if not rows:
        msg = f"CSV {path.name} に有効な気温データがありません"
        raise ValueError(msg)
    return rows


def format_temperature_series(series: Sequence[tuple[str, float]]) -> str:
    lines = ["日付,気温(°C)"]
    lines.extend(f"{label},{temperature}" for label, temperature in series)
    return "\n".join(lines)


def analyze_text(
    text: str,
    prompt: str = PRESET_PROMPT,
    llm: GraphLanguageModel | None = None,
) -> str:
    try:
        model = llm or GeminiGraphLanguageModel.from_env()
    except Exception as exc:
        return f"{prompt}\n- Gemini呼び出しに失敗しました: {exc}"

    try:
        return model.comment_on_text(text, prompt)
    except Exception as exc:
        return f"{prompt}\n- Gemini呼び出しに失敗しました: {exc}"


def analyze_image(
    image: Image.Image,
    prompt: str = PRESET_PROMPT,
    llm: GraphLanguageModel | None = None,
) -> str:
    try:
        model = llm or GeminiGraphLanguageModel.from_env()
    except Exception as exc:
        return f"{prompt}\n- Gemini呼び出しに失敗しました: {exc}"

    try:
        return model.comment_on_graph(image, prompt)
    except Exception as exc:
        return f"{prompt}\n- Gemini呼び出しに失敗しました: {exc}"
