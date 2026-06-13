# Chạy toàn bộ model bằng Hugging Face Space

## Kiến trúc

- Vercel tính 12 feature cho trận đấu.
- Hugging Face Docker Space tải model được chọn và chạy inference.
- Vercel gọi `POST /predict` rồi trả kết quả cho giao diện.

## Tạo Space

1. Trên Hugging Face, tạo Space mới với SDK `Docker`.
2. Clone repository của Space về máy.
3. Chạy:

```powershell
.\hf_space\prepare_models.ps1
```

4. Sao chép toàn bộ nội dung bên trong `hf_space` vào repository Space.
5. Trong repository Space, bật Git LFS rồi push:

```powershell
git lfs install
git lfs track "models/*.pkl"
git add .
git commit -m "Deploy WC 2026 model inference"
git push
```

## Bảo vệ API

Trong Settings của Space, tạo secret:

```text
INFERENCE_API_TOKEN=<một-token-bí-mật-dài>
```

## Kết nối Vercel

Trong project `wc-model` trên Vercel, tạo hai Environment Variables:

```text
INFERENCE_API_URL=https://<username>-<space-name>.hf.space
INFERENCE_API_TOKEN=<cùng-token-ở-trên>
```

Redeploy Vercel. API `/api/thuat-toan` sẽ tự đánh dấu cả bốn model là khả dụng và chuyển inference sang Hugging Face.

## API của Space

```http
GET /health
GET /models
POST /predict
Authorization: Bearer <INFERENCE_API_TOKEN>
```

Hugging Face Space CPU miễn phí có thể sleep khi không dùng. Request đầu tiên sau thời gian nghỉ sẽ chậm hơn do cold start.
