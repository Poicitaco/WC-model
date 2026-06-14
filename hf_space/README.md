---
title: WC 2026 Model Inference
emoji: ⚽
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# WC 2026 Model Inference

Docker Space phục vụ inference cho bốn model:

- `random_forest_baseline`
- `random_forest_tuned`
- `xgboost_baseline`
- `xgboost_tuned`

API:

- `GET /health`
- `GET /models`
- `POST /predict`

Đặt biến bí mật `INFERENCE_API_TOKEN` trong Space để bảo vệ API. Vercel cần dùng cùng token.
