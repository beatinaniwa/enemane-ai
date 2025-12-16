from pathlib import Path

import pytest
from google.genai import types as genai_types
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pytest import MonkeyPatch

from enemane_ai import analyzer
from enemane_ai.analyzer import (
    PRESET_PROMPT,
    DailyPowerSummary,
    MonthlyPowerCalendarData,
    MonthlyReportData,
    MonthlyTemperatureSummary,
    analyze_image,
    analyze_text,
    build_power_calendar_context,
    build_supplementary_context,
    collect_graph_entries,
    parse_monthly_report_csv,
    parse_power_30min_csv,
    parse_temperature_csv_for_comparison,
)


class DummyLLM:
    def __init__(self, response: str = "analysis"):
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_text: str | None = None

    def comment_on_graph(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response

    def comment_on_text(self, text: str, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_text = text
        return self.response


def test_analyze_image_mentions_brightness_levels() -> None:
    llm = DummyLLM("LLM response")
    image = Image.new("RGB", (24, 24), "white")

    comment = analyze_image(image, prompt="custom prompt", llm=llm)

    assert comment == "LLM response"
    assert llm.calls == 1
    assert llm.last_prompt == "custom prompt"


def test_analyze_text_uses_llm() -> None:
    llm = DummyLLM("text response")

    comment = analyze_text("2024-01-01,8.5", prompt="csv prompt", llm=llm)

    assert comment == "text response"
    assert llm.calls == 1
    assert llm.last_prompt == "csv prompt"
    assert llm.last_text == "2024-01-01,8.5"


def test_analyze_text_does_not_leak_prompt_on_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = analyze_text("text content", prompt="top-secret prompt")

    assert "top-secret prompt" not in result
    assert "失敗しました" in result


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
    assert all(entry.image is not None and entry.image.size[0] > 0 for entry in entries)


def test_collect_graph_entries_from_temperature_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "temperature.csv"
    csv_path.write_text(
        "date,temp_c\n" "2024-01-01,8.5\n" "2024-01-02,12.3\n" "2024-01-03,5.0\n",
        encoding="utf-8",
    )

    entries = collect_graph_entries([csv_path])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.display_label == "temperature.csv"
    assert entry.image is None
    assert entry.text is not None
    assert "2024-01-02,12.3" in entry.text


def test_collect_graph_entries_from_shift_jis_temperature_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "maebashi_shift_jis.csv"
    csv_path.write_bytes(
        (
            "ダウンロードした時刻：2025/10/08 12:19:10\r\n"  # noqa: RUF001
            "\r\n"
            ",前橋,前橋\r\n"
            "年月日時,気温(℃),品質情報\r\n"
            "2023/10/1 1:00:00,22.9,8\r\n"
            "2023/10/1 2:00:00,22.3,8\r\n"
        ).encode("cp932")
    )

    entries = collect_graph_entries([csv_path])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.display_label == "maebashi_shift_jis.csv"
    assert entry.image is None
    assert entry.text is not None
    assert "2023/10/1 1:00:00,22.9" in entry.text
    assert "2023/10/1 2:00:00,22.3" in entry.text


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

    text_comment = model.comment_on_text("raw csv content", PRESET_PROMPT)
    assert text_comment == "Gemini says hello"
    assert calls.contents == [PRESET_PROMPT, "raw csv content"]


def test_parse_monthly_report_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "月報202310.csv"
    csv_path.write_text(
        "(1) 参照元「月間エネルギー使用実績と最大電力の推移」,,,\n"
        ",10/1(日),10/2(月),10/3(火)\n"
        "最大電力[kW],27.52,35.14,34.18\n"
        "1F事務所SR_電灯,100.47,97,99.53\n"
        "受電電力,330.95,465.97,458.73\n",
        encoding="utf-8",
    )

    report = parse_monthly_report_csv(csv_path)

    assert report.month_label == "2023年10月"
    assert report.max_power_daily == [27.52, 35.14, 34.18]
    assert report.max_power_monthly == 35.14
    assert report.total_power_daily == [330.95, 465.97, 458.73]
    assert report.total_power_monthly == 1255.65
    assert "1F事務所SR_電灯" in report.circuits
    assert report.circuits["1F事務所SR_電灯"] == [100.47, 97, 99.53]


def test_parse_temperature_csv_for_comparison(tmp_path: Path) -> None:
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2023/10/1 2:00:00,20.0,8\n"
        "2024/10/1 1:00:00,25.0,8\n"
        "2024/10/1 2:00:00,23.0,8\n",
        encoding="utf-8",
    )

    prev, curr = parse_temperature_csv_for_comparison(csv_path)

    assert prev.year_month == "2023-10"
    assert prev.max_temp == 22.9
    assert prev.min_temp == 20.0
    assert prev.avg_temp == 21.45

    assert curr.year_month == "2024-10"
    assert curr.max_temp == 25.0
    assert curr.min_temp == 23.0
    assert curr.avg_temp == 24.0


def test_parse_temperature_csv_for_comparison_with_next_month_data(
    tmp_path: Path,
) -> None:
    """月末翌日のデータ(1件)が含まれる場合でも同じ月を正しく比較する."""
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2023/10/1 2:00:00,20.0,8\n"
        "2023/11/1 0:00:00,14.2,8\n"  # 月末翌日データ(1件)
        "2024/10/1 1:00:00,25.0,8\n"
        "2024/10/1 2:00:00,23.0,8\n"
        "2024/11/1 0:00:00,13.1,8\n",  # 月末翌日データ(1件)
        encoding="utf-8",
    )

    prev, curr = parse_temperature_csv_for_comparison(csv_path)

    # 同じ月(10月)を比較すること
    assert prev.year_month == "2023-10"
    assert curr.year_month == "2024-10"
    # 11月の1件データは無視される
    assert prev.max_temp == 22.9
    assert curr.max_temp == 25.0


def test_parse_temperature_csv_for_comparison_no_matching_month(
    tmp_path: Path,
) -> None:
    """同じ月の前年・当年データがない場合はエラー."""
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2024/11/1 1:00:00,25.0,8\n",  # 異なる月
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="同じ月の前年・当年データが見つかりません"):
        parse_temperature_csv_for_comparison(csv_path)


def test_build_supplementary_context_with_data() -> None:
    report = MonthlyReportData(
        month_label="2023年10月",
        max_power_daily=[30.0, 35.0, 32.0],
        circuits={"1F電灯": [100.0, 110.0, 105.0], "2F電灯": [50.0, 55.0, 52.0]},
        total_power_daily=[300.0, 350.0, 320.0],
    )
    prev = MonthlyTemperatureSummary(
        year_month="2023-10", max_temp=27.0, min_temp=10.0, avg_temp=18.0
    )
    curr = MonthlyTemperatureSummary(
        year_month="2024-10", max_temp=30.0, min_temp=12.0, avg_temp=20.0
    )

    context = build_supplementary_context(report, (prev, curr))

    assert "2023年10月" in context
    assert "35.0 kW" in context
    assert "970 kWh" in context
    assert "1F電灯" in context
    assert "2024年10月" in context
    assert "+3.0℃" in context
    assert "+2.0℃" in context


def test_build_supplementary_context_with_none() -> None:
    context = build_supplementary_context(None, None)
    assert context == ""


def test_parse_power_30min_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "PDU_30min_202410.csv"
    csv_path.write_bytes(
        (
            "計測対象,コントローラ1,,,,,,,\n"
            ",機器31-1,,,,,,,,\n"
            ",受電電力量,,,,,,,,\n"
            ",kWh,,,,,,,,\n"
            "2024-10-01 00:00,4.29,,,,,,,\n"
            "2024-10-01 00:30,4.04,,,,,,,\n"
            "2024-10-01 01:00,4.08,,,,,,,\n"
            "2024-10-05 00:00,3.50,,,,,,,\n"  # 土曜日
            "2024-10-05 00:30,3.60,,,,,,,\n"
            "2024-10-06 00:00,3.20,,,,,,,\n"  # 日曜日
            "2024-10-06 00:30,3.30,,,,,,,\n"
        ).encode("cp932")
    )

    data = parse_power_30min_csv(csv_path)

    assert data.year_month == "2024年10月"
    assert len(data.daily_summaries) == 3
    # 10/1のデータ確認
    day1 = data.daily_summaries[0]
    assert day1.date == "2024-10-01"
    assert day1.day_of_week == "火"
    assert day1.total_kwh == 4.29 + 4.04 + 4.08
    assert day1.max_kwh == 4.29
    assert day1.max_time == "00:00"
    # 最大電力使用量日 (1日の合計が最大 = 10/1)
    assert "1日" in data.max_usage_day
    assert data.max_usage_kwh == 4.29 + 4.04 + 4.08
    # 最大需要電力日 (30分ピークが最大 = 10/1の4.29kW)
    assert "1日" in data.max_demand_day
    assert data.max_demand_kw == 4.29
    # 平日/休日平均
    assert data.weekday_avg_kwh > 0
    assert data.weekend_avg_kwh > 0


def test_parse_power_30min_csv_no_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("header1,header2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="有効な30分電力データがありません"):
        parse_power_30min_csv(csv_path)


def test_build_power_calendar_context() -> None:
    data = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[
            DailyPowerSummary(
                date="2024-10-01",
                day_of_week="火",
                total_kwh=300.0,
                max_kwh=21.5,
                max_time="14:00",
            ),
            DailyPowerSummary(
                date="2024-10-05",
                day_of_week="土",
                total_kwh=100.0,
                max_kwh=8.0,
                max_time="10:00",
            ),
        ],
        total_monthly_kwh=8500.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=290.0,
        weekend_avg_kwh=95.0,
    )

    context = build_power_calendar_context(data)

    assert "2024年10月" in context
    assert "8,500.0 kWh" in context
    assert "最大電力使用量日: 1日(火)" in context
    assert "300.0 kWh" in context
    assert "最大需要電力日: 1日(火)" in context
    assert "21.5 kW" in context
    assert "290.0 kWh/日" in context
    assert "95.0 kWh/日" in context
    assert "上位5日" in context
