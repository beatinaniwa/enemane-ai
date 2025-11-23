# enemane-ai

グラフ画像 / PDF / 気温CSV をまとめてアップロードし、決められたプロンプトに基づいてコメントを返す Streamlit アプリです。パッケージ管理は `uv` を使用し、Python 3.13 を想定しています。グラフの要約には Google Gemini (gemini-3-pro-preview) を利用します。

## セットアップ

1. Python 3.13 系を用意してください（`uv python install 3.13` など）。
2. 依存関係を同期:
   ```bash
   uv sync --dev
   ```
3. Google Gemini API キーを環境変数に設定:
   ```bash
   export GEMINI_API_KEY="your-key"
   ```
   Streamlit の `secrets.toml` (`.streamlit/secrets.toml`) に `GEMINI_API_KEY` を置いても構いません。
4. pre-commit をインストール:
   ```bash
   uv run pre-commit install
   ```

## 開発とテスト (TDD)

- 仕様をテストで落とし込んでから実装を追加する t-wada 推奨の TDD 手順を採用しています。
- テスト実行:
  ```bash
  uv run pytest
  ```

## アプリの起動

```bash
uv run streamlit run src/enemane_ai/app.py
```

CSV をアップロードする場合は、`日付,気温` の2列構成を想定しています (例: `2024-01-01,12.3`)。
CSV はグラフ化せず、そのままテキストとして解析に渡されます。

## コード品質

- `ruff` による Lint/フォーマットを pre-commit で自動実行します。
- 手動で実行する場合:
  ```bash
  uv run ruff check
  uv run ruff format
  ```

## Git / GitHub (gh)

- リポジトリ初期化とリモート作成例:
  ```bash
  git init
  gh repo create --public
  git add .
  git commit -m "Add Streamlit batch analyzer scaffold"
  git push -u origin main
  ```
