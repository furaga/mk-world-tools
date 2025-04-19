# mk-world-tools
マリオカートワールドの自動集計などのツール群

## 動作環境

- Windows 11
- Python 3.12.5
- uv 0.6.14 (a4cec56dc 2025-04-09)

## セットアップ

```bash
uv pip sync pyproject.toml
```

## Tools

### screenshot_script.py

動画URLのリストから指定されたタイムスタンプのフレームを抽出するスクリプトです。

#### セットアップ

1.  **yt-dlp の準備:**
    *   [yt-dlpのGitHubリリースページ](https://github.com/yt-dlp/yt-dlp/releases/latest) から、Windows向けの `yt-dlp.exe` をダウンロードします。
    *   ダウンロードした `yt-dlp.exe` を `tools/third_party/` ディレクトリに配置します。
        ```
        mk-world-tools/
        └── tools/
            ├── third_party/
            │   └── yt-dlp.exe  <-- ここに配置
            └── screenshot_script.py
        ```

#### 使い方

1.  **URLリストファイルの準備:**
    *   フレームを抽出したい動画のURLとタイムスタンプ（秒）を記載したテキストファイルを作成します。
    *   URLには `?t=<秒数>` または `&t=<秒数>` の形式でタイムスタンプを追加します。
    *   1行に1つのURLを記載します。
    *   `#` で始まる行はコメントとして無視されます。

    例 (`urls.txt`):
    ```
    # 例: YouTube動画の特定の時間
    https://www.youtube.com/watch?v=XXXXXXXXXXX&t=123s
    https://youtu.be/YYYYYYYYYYY?t=45
    # タイムスタンプが整数でない場合も可
    https://example.com/video.mp4?t=67.8
    ```

2.  **スクリプトの実行:**
    *   コマンドプロンプトやターミナルで以下のコマンドを実行します。

    ```bash
    uv run python tools/screenshot_script.py --url-file <URLリストファイルのパス> [--output-dir <出力先ディレクトリ>] [--max-workers <並列処理数>] [--yt-dlp-path <yt-dlpのパス>]
    ```

    *   **必須引数:**
        *   `--url-file <URLリストファイルのパス>`: 上記で作成したURLリストファイルへのパスを指定します。
    *   **オプション引数:**
        *   `--output-dir <出力先ディレクトリ>`: スクリーンショットを保存するディレクトリを指定します (デフォルト: `screenshots`)。
        *   `--max-workers <並列処理数>`: ダウンロードとフレーム抽出を並列で行う最大ワーカー数を指定します (デフォルト: `4`)。
        *   `--yt-dlp-path <yt-dlpのパス>`: `yt-dlp.exe` へのパスを指定します (デフォルト: `tools/third_party/yt-dlp.exe`)。

    実行例:
    ```bash
    uv run python tools/screenshot_script.py --url-file urls.txt --output-dir captured_frames --max-workers 8
    ```

    これにより、`urls.txt` に記載された各URLの指定時間に対応するフレーム画像が `captured_frames` ディレクトリに保存されます。ファイル名は、URLの一部とタイムスタンプから自動生成されます (例: `watch_v=XXXXXXXXXXX_0203-00.jpg`)。
