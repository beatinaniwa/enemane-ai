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
from PIL import Image, ImageDraw, ImageFont

PRESET_PROMPT = (
    "あなたはグラフリテラシーに長けたアナリストです。"
    " 以下のフォーマットで簡潔にコメントを返してください:"
    " 1) トレンド 2) 読み取れる含意 3) 注意点。"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
CSV_EXTENSIONS = {".csv"}
CSV_CHART_SIZE = (900, 520)
DEFAULT_MODEL_NAME = "gemini-3-pro-preview"


@dataclass
class GraphEntry:
    display_label: str
    image: Image.Image


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
    image = render_temperature_chart(series, title=path.stem)
    return GraphEntry(display_label=path.name, image=image)


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


def render_temperature_chart(series: Sequence[tuple[str, float]], title: str) -> Image.Image:
    width, height = CSV_CHART_SIZE
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 60, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    temperatures = [item[1] for item in series]
    min_temp = min(temperatures)
    max_temp = max(temperatures)
    if max_temp == min_temp:
        min_temp -= 1
        max_temp += 1
    padding = (max_temp - min_temp) * 0.05
    min_temp -= padding
    max_temp += padding

    def y_for(value: float) -> float:
        normalized = (value - min_temp) / (max_temp - min_temp)
        return height - margin_bottom - (normalized * plot_height)

    step = plot_width / max(len(series) - 1, 1)
    points = [(margin_left + index * step, y_for(temp)) for index, (_, temp) in enumerate(series)]

    draw.rectangle(
        (
            margin_left,
            margin_top,
            width - margin_right,
            height - margin_bottom,
        ),
        outline="#d0d7de",
        width=1,
    )
    draw.line(
        (
            margin_left,
            height - margin_bottom,
            width - margin_right,
            height - margin_bottom,
        ),
        fill="black",
        width=2,
    )
    draw.line(
        (
            margin_left,
            height - margin_bottom,
            margin_left,
            margin_top,
        ),
        fill="black",
        width=2,
    )

    draw.text((margin_left, 20), f"{title} の気温推移", fill="black", font=font)
    draw.text((10, margin_top - 10), f"最高 {max_temp:.1f}°C", fill="gray", font=font)
    draw.text(
        (10, height - margin_bottom - 10),
        f"最低 {min_temp:.1f}°C",
        fill="gray",
        font=font,
    )

    for index, (label, _) in enumerate(series):
        if index not in {0, len(series) // 2, len(series) - 1}:
            continue
        x = margin_left + (index * step if len(series) > 1 else plot_width / 2)
        draw.text(
            (x - 10, height - margin_bottom + 8),
            label,
            fill="gray",
            font=font,
        )

    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1e88e5", outline="#1e88e5")
        return image

    draw.line(points, fill="#1e88e5", width=3)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1e88e5", outline="#1e88e5")

    return image


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
