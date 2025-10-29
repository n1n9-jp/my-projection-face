# projection-face ツール開発メモ

このリポジトリでは、ウェブカメラで撮影した顔写真を線画化し、GeoJSONとして出力したデータを `projection-face` に取り込むツールの開発を進める。

## 実装プラン（初期案）

1. **仕様整理**  
   - 入出力形式、想定解像度、線画化の品質要件、GeoJSONの属性構造を箇条書きで明文化する。  
   - パイプライン全体（カメラ入力 → 顔検出・前処理 → 線画変換 → SVGベクタ化 → GeoJSON変換 → projection-face取り込み → Raycast連携）のインターフェースを整理する。
2. **線画化～GeoJSON化のPoC**  
   - 既存の静止画サンプルで線画化アルゴリズムとSVGベクタ化を試し、品質と手順を確かめる。  
   - SVGパスをGeoJSONのジオメトリ（`LineString`/`Polygon`）に変換するロジックを試作し、座標系・縮尺の取り扱いを検証する。
3. **CLIパイプライン試作**  
   - コマンドラインから `python pipeline.py --input sample.jpg --output face.geojson` のように実行し、GeoJSONが得られる最小パイプラインを実装する。  
   - エラーハンドリングやログ出力の基本形を整える。
4. **projection-face 連携**  
   - 既存プロジェクトのデータ取り込みポイントを確認し、生成したGeoJSONを組み込める API/モジュールを実装する。  
   - サンプルGeoJSONで描画結果を確認し、必要な座標変換やメタデータを調整する。
5. **Raycast AI Extensions 連携**  
   - Raycastの拡張を作成し、CLIパイプラインを呼び出す仕組み（入力UI、実行、結果表示）を組み込む。  
   - 権限や実行環境（カメラへのアクセス許可など）を整理する。
6. **テストとドキュメント整備**  
   - 線画化やGeoJSON変換のユニットテスト、サンプル画像を使ったエンドツーエンドテストを準備する。  
   - 利用手順、必要ツール、注意点を README に追記して、再現性を高める。

## 実現性の評価と懸念点

- **実現性**: 各工程は既存ライブラリやツールでカバー可能であり、段階的に進めれば十分実現可能。小さなマイルストーンを設定し、PoCでリスクを抑える方針が有効。  
- **懸念点**:  
  - 線画化の品質とパフォーマンスが最難関。輪郭抽出・ノイズ除去のチューニングが必要。  
  - SVG→GeoJSON 変換時の座標系・縮尺・精度の扱いが不明確だと描画にズレが発生する恐れ。  
  - Raycast連携は最後のフェーズで検討し、ドキュメントを十分に確認してから実装する。  
  - リアルタイム性が要件の場合、処理速度の最適化が追加で必要。

## 次のステップ（予定）

1. 仕様の詳細メモとサンプルデータの準備。  
2. SVG→GeoJSON 変換ロジックの検証（サンプルSVGを用意し、ジオメトリ生成を確認）。  
3. 線画化～ベクタ化のPoCを実施し、GeoJSON変換との結合テストを進める。

## SVG→GeoJSON 変換仕様（初期案）

- **入力**: SVG 1.1 テキスト。対象要素は `path`, `polyline`, `polygon`。`path` コマンドは `M`, `L`, `H`, `V`, `Z`（絶対・相対）に対応。  
- **前提**: 曲線コマンドや相対座標は線画化フェーズでポリライン化しておく。スタイル情報やグループは現時点では無視。  
- **座標系**: SVG 側ではピクセル座標（原点は左上、Y は下方向）を前提に解析し、必要に応じて出力時に経度/緯度範囲へ正規化する（`--no-normalize` で無効化可能）。  
- **出力形式**: GeoJSON FeatureCollection。各 SVG 要素を 1 Feature 化し、`properties` に `type`・`id` などのメタ情報を格納。  
- **ジオメトリ規則**: `path`/`polyline` は `LineString`、閉路は `Polygon` に昇格。`polygon` は `Polygon` とし、最初と最後の座標は重複させない。  
- **エラーハンドリング**: 非対応コマンドに遭遇したらログ出力して終了コード 1。座標欠損も同様。将来的には `--ignore-unsupported` を検討。  
- **検証データ**: `face_simple.svg` を用意し、変換結果を `face_simple.geojson` として目視確認する。

## SVG→GeoJSON 変換ツール（現状）

- `svg_to_geojson.py` を追加。`path`/`polyline`/`polygon` を読み取り、対応する GeoJSON FeatureCollection を生成する。  
- 標準では SVG 座標を全体のバウンディングボックスに合わせて経度/緯度範囲（デフォルト: `[-179, 179]`, `[-85, 85]`）に線形変換するため、geojson.io などのビューアで読み込める。元のピクセル座標を保ちたい場合は `--no-normalize` を指定する。  
- 正規化範囲は `--normalize-range LON_MIN LON_MAX LAT_MIN LAT_MAX` で変更可能。Web Mercator に合わせる場合は `--normalize-range -179 179 -85 85` 程度が安全。  
- メタデータ（元座標のバウンディングボックスなど）を含めたいときは `--include-metadata` を併用する。  
- 利用例: `python3 svg_to_geojson.py --input face_simple.svg --output face_simple.geojson`  
- 今後のTODO: 非対応コマンドのスキップオプション、スタイル/グループ階層の扱い、正規化範囲のプリセット追加、座標精度調整を検討する。

## 顔線画化 PoC（OpenCV版・保留中）

- `experiments/opencv_face_lineart/face_lineart_poc.py` に OpenCV ベースのパイプライン（顔検出 → 前処理 → Canny エッジ → 輪郭抽出 → SVG 化）を試作。CLAHE + auto Canny（二段階）+ thinning（`opencv-contrib-python` があれば有効）で線の滑らかさを向上。  
- 依存ライブラリや調整結果は `experiments/opencv_face_lineart/face_lineart_notes.md` に記録。  
- 現在は Stable Diffusion + ControlNet を利用する方向へ移行予定のため、このOpenCV版はアーカイブ扱い。必要になった場合に参照する。  

## projection-face 連携テスト（2024-XX-XX）

- リポジトリ: `Projects_DataViz_SelfWorks/projection-face`（`origin/main`）。  
- 手順: 既存の `face.geojson` をバックアップ後、生成した `face_simple.geojson` をコピーし、簡易バリデーション（FeatureCollectionかつ LineString/Polygon のみで構成）を Python スクリプトで確認。  
- 結果: 11 フィーチャーを含む `FeatureCollection` として正常に読み込まれ、geojson.io でも表示可能な範囲に正規化されていることを確認。  
- 備考: 元ファイルは `face.geojson.bak` として保存。実描画確認はブラウザで `index.html` を開いて行う。
