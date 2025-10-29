# ControlNet LineArt PoC

Stable Diffusion 1.5 + ControlNet(LineArt) を用いてローカル環境 (Apple Silicon, MPS) で線画抽出を試すためのメモ。

## セットアップ

```bash
python3 -m venv .sd-venv
source .sd-venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate safetensors controlnet-aux
```

- PyTorch は MPS バックエンドを自動的に利用する構成。  
- 初回実行時は Hugging Face から `runwayml/stable-diffusion-v1-5` と `lllyasviel/control_v11p_sd15_lineart` をダウンロードするため数分かかる。

## 実行例

```bash
source .sd-venv/bin/activate
python experiments/controlnet_lineart/lineart_poc.py \
  --image samples/lena.jpg \
  --output-dir outputs/controlnet \
  --steps 15 \
  --guidance-scale 7.0
```

- `outputs/controlnet/lena_control_lineart.png`: LineartDetector による線画抽出結果  
- `outputs/controlnet/lena_generated.png`: Stable Diffusion + ControlNet によって生成された線画

## 所感 / TODO

- M4 MacBook Air (MPS) で 512x512, 15 steps の場合、生成に ~20 秒程度。  
- 出力された線画は OpenCV ベースよりも陰影の表現が豊富で、髪のラインも滑らか。  
- 今後の課題:
  - `potrace` 等でのベクタ化 → `svg_to_geojson.py` への接続
  - プロンプト/ガイダンススケールのチューニング
  - 自前顔写真（`samples/self.jpg`）での画質評価とパラメータ最適化
