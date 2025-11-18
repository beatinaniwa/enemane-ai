from pathlib import Path

from google.genai import types as genai_types
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pytest import MonkeyPatch

from enemane_ai import analyzer
from enemane_ai.analyzer import PRESET_PROMPT, analyze_image, collect_graph_entries


class DummyLLM:
    def __init__(self, response: str = "analysis"):
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None

    def comment_on_graph(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response


def test_analyze_image_mentions_brightness_levels() -> None:
    llm = DummyLLM("LLM response")
    image = Image.new("RGB", (24, 24), "white")

    comment = analyze_image(image, prompt="custom prompt", llm=llm)

    assert comment == "LLM response"
    assert llm.calls == 1
    assert llm.last_prompt == "custom prompt"


def test_collect_graph_entries_from_png_and_pdf(tmp_path: Path) -> None:
    png_path = tmp_path / "plot.png"
    base = Image.new("RGB", (80, 60), "white")
    draw = ImageDraw.Draw(base)
    draw.line((10, 50, 70, 10), fill="black", width=3)
    base.save(png_path)

    pdf_path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    with pdf_path.open("wb") as fp:
        writer.write(fp)

    entries = collect_graph_entries([png_path, pdf_path])

    labels = {entry.display_label for entry in entries}
    assert len(entries) == 2
    assert any("plot.png" in label for label in labels)
    assert any("doc.pdf#1" in label for label in labels)
    assert all(entry.image.size[0] > 0 for entry in entries)


def test_gemini_model_uses_env_key(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    class CallLog:
        def __init__(self) -> None:
            self.api_key: str | None = None
            self.model: str | None = None
            self.contents: list[genai_types.Part | str] | None = None

    calls = CallLog()

    class FakeModelAPI:
        @staticmethod
        def generate_content(model: str, contents: list[genai_types.Part | str]) -> object:
            calls.model = model
            calls.contents = contents

            class Response:
                text = "Gemini says hello"

            return Response()

    class FakeClient:
        def __init__(self, api_key: str):
            calls.api_key = api_key

        models = FakeModelAPI()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(analyzer, "genai", type("GenaiWrapper", (), {"Client": FakeClient}))

    model = analyzer.GeminiGraphLanguageModel.from_env()

    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), "black").save(img_path)
    with Image.open(img_path) as img:
        comment = model.comment_on_graph(img, PRESET_PROMPT)

    assert comment == "Gemini says hello"
    assert calls.api_key == "fake-key"
    assert calls.model == analyzer.DEFAULT_MODEL_NAME
    assert calls.contents is not None
    assert len(calls.contents) == 2
    assert calls.contents[0] == PRESET_PROMPT
    image_part = calls.contents[1]
    assert isinstance(image_part, genai_types.Part)
    inline_data = image_part.inline_data
    assert inline_data is not None
    assert inline_data.mime_type == "image/png"
