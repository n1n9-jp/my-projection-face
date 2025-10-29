# 顔線画化 PoC メモ

## 実験概要

- 目的: OpenCV を用いた「顔検出 → 前処理 → エッジ抽出 → 簡易ベクタ化」の手順を検証し、基礎的なワークフローの実現性を確認する。  
- 使用スクリプト: `poc/face_lineart_poc.py`  
- 依存ライブラリ: `opencv-python`, `numpy`（`python3 -m venv .venv` の仮想環境で `pip install opencv-python`）  
- サンプル画像: `samples/lena.jpg`（OpenCV サンプル画像）

## 実行コマンド

```bash
source .venv/bin/activate
python poc/face_lineart_poc.py --image samples/lena.jpg --output-dir outputs
```

## 出力結果

- `outputs/lena_crop.png` : 顔領域を余白 20% で切り出したカラー画像  
- `outputs/lena_edges.png` : 前処理後に Canny エッジを適用した 1bit 画像  
- `outputs/lena_contours.svg` : エッジを `cv2.findContours` で抽出し、ポリラインとして SVG 化したもの  
- メトリクス（`outputs/lena_edges.png`）: エッジ画素 4,910（全体の約 8.9%）、抽出された輪郭数 9

### フォールバック動作

- 生成テスト用の `samples/test_face.png`（OpenCV で合成したシンプルな顔）では Haar カスケードで検出できないため、フォールバックとして画像全体を処理。  
- コマンド例: `python poc/face_lineart_poc.py --image samples/test_face.png --output-dir outputs`  
- 結果: `[WARN] 顔が検出できませんでした… -> 画像全体を使用します。` のメッセージを出しつつ処理継続。輪郭数は 1（外形のみ）。  
- フォールバックを無効化したい場合は `--no-fallback` を指定。

## 処理フロー

1. `haarcascade_frontalface_default.xml` による顔検出。最大のバウンディングボックスを利用。  
2. 顔領域を切り出し、ヒストグラム平坦化 + bilateral フィルタで平滑化。  
3. Canny(40, 120) でエッジ抽出 → 3x3 カーネルで膨張し、外形を繋げる。  
4. `cv2.findContours` → `cv2.approxPolyDP` で輪郭を単純化し、SVG の `path` 要素に変換。  

## 気付き・今後の改善ポイント

- 顔のパーツ（目・鼻・口）の輪郭は抽出できたが、髪や陰影の線は抽象度が高く、さらなる調整（スムージングや閾値調整）が必要。  
- 線の太さ・密度は Canny の閾値や前処理のパラメータに大きく依存する。被写体に合わせた自動調整ロジックを検討したい。  
- `cv2.findContours` ベースのベクタ化は出力がギザギザになりがち。`potrace` など外部ツールで滑らかな曲線化を比較したい。  
- 顔検出に失敗した場合のフォールバック（全体処理・手動トリミング）や、複数顔がある場合の扱いを検討する必要がある。  
- 今回の SVG 出力は座標がピクセル単位のまま。将来的に GeoJSON へ接続する際は、正規化やスケール調整を組み込みたい。
