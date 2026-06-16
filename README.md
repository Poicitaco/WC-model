# Du doan World Cup 2026 bang AI

Ung dung web Flask dung de phan tich, mo phong va du doan World Cup 2026 bang du lieu lich su, ranking FIFA, Elo va cac ket qua mo phong san. Du an co giao dien chinh, API du doan tran dau, lich thi dau, bang dau, knockout, che do thuyet trinh va photobooth chu de World Cup.



## Tinh nang chinh

- Du doan ket qua tran dau theo doi A, doi B va thuat toan duoc chon.
- Xem du doan vong bang, bang xep hang, top doi di tiep va nhanh knockout.
- Hien thi cac bieu do danh gia mo hinh trong `static/anh`.
- Doc du lieu CSV da xu ly trong `data_processed` va ket qua mo phong trong `outputs`.
- Trang thuyet trinh HTML tai `/thuyettrinh`.
- Photobooth tai `/photobooth` va trang quan tri khung anh an.
- Ho tro deploy Vercel thong qua `vercel.json`.

## Cong nghe

- Backend: Python, Flask, pandas, numpy, joblib.
- Frontend: HTML, CSS, JavaScript.
- Runtime Vercel: `@vercel/python`.
- Static files: thu muc `static`.
- Du lieu: CSV trong `data_processed` va `outputs`.

## Cau truc du an

```text
WC2026_FINAL/
|-- app.py                         # Flask app, routes va API
|-- requirements.txt               # Thu vien Python can cai
|-- vercel.json                    # Cau hinh deploy Vercel
|-- .vercelignore                  # File/thu muc khong upload len Vercel
|-- templates/                     # HTML templates
|   |-- index.html
|   |-- photobooth.html
|   `-- admin_photobooth.html
|-- static/                        # CSS, anh, presentation, khung photobooth
|-- data_processed/                # Du lieu CSV da xu ly
|-- outputs/                       # Ket qua du doan/mo phong da xuat san
|-- models/                        # Model local .pkl, bi ignore khi deploy Vercel
`-- hf_space/                      # Ban Docker/Hugging Face Space rieng
```

## Chay local truoc khi deploy

### 1. Cai Python

Khuyen dung Python 3.10 hoac 3.11. Kiem tra bang:

```bash
python --version
```

### 2. Tao moi truong ao

Tren Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Tren macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Cai dependencies

```bash
pip install -r requirements.txt
```

### 4. Chay ung dung

```bash
python app.py
```

Mo trinh duyet tai:

```text
http://127.0.0.1:5000
```

Mot so duong dan can test:

- `/` - trang chinh
- `/photobooth` - photobooth
- `/thuyettrinh` - slide thuyet trinh
- `/api/danh-sach-doi` - API danh sach doi
- `/api/thuat-toan` - API danh sach thuat toan

## Bien moi truong

Ung dung co the chay khong can bien moi truong, nhung khi deploy that nen cau hinh cac bien sau trong Vercel.

| Bien | Bat buoc | Gia tri goi y | Y nghia |
| --- | --- | --- | --- |
| `ADMIN_PASSWORD` | Khuyen dung | Mat khau manh cua ban | Mat khau trang quan tri photobooth. Neu khong set, app dung mac dinh trong code. |
| `ADMIN_PATH` | Khuyen dung | `quan-tri-khung-anh-<chuoi-bi-mat>` | Duong dan trang admin photobooth. |
| `INFERENCE_API_URL` | Khuyen dung | URL Hugging Face/API inference | API du doan tu xa. Mac dinh code dang tro den Hugging Face Space co san. |
| `INFERENCE_API_TOKEN` | Tuy chon | Token API neu endpoint rieng can auth | Gui qua header `Authorization: Bearer ...`. |
| `FOOTBALL_API_TOKEN` | Tuy chon | Token tu football-data.org | Lay lich thi dau chinh thuc tu Football-Data API. Khong co token thi app dung du lieu local/fallback. |
| `BLOB_READ_WRITE_TOKEN` | Tuy chon | Token Vercel Blob | Chi can neu muon upload/xoa khung photobooth tren Vercel. |

Luu y: Khong commit token that vao GitHub. Hay them trong Vercel Dashboard.

## Luu y quan trong ve model khi deploy Vercel

Thu muc `models/` dang nam trong `.vercelignore`, vi vay cac file `.pkl` se khong duoc dua len Vercel. Cach nay giup deployment nhe hon va tranh loi dung luong function.

Voi cau hinh hien tai, du doan dong se uu tien goi `INFERENCE_API_URL`. Neu endpoint tu xa bi loi, app co co che fallback local, nhung tren Vercel fallback local co the that bai vi khong co file model.

Neu ban muon deploy Vercel on dinh, nen lam mot trong hai cach:

1. Giu `models/` trong `.vercelignore` va cau hinh `INFERENCE_API_URL` toi mot API inference rieng.
2. Bo `models/` khoi `.vercelignore` chi khi chac chan tong dung luong nam trong gioi han Vercel va dependencies co the load model thanh cong.

Khuyen dung cach 1.

## Deploy len Vercel bang GitHub

### 1. Day source len GitHub

Neu chua co repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Neu da co repository, chi can commit va push:

```bash
git add .
git commit -m "Update project"
git push
```

Truoc khi push, dam bao cac thu muc sau co trong repository:

- `app.py`
- `templates/`
- `static/`
- `data_processed/`
- `outputs/`
- `requirements.txt`
- `vercel.json`

### 2. Import project tren Vercel

1. Dang nhap https://vercel.com.
2. Chon **Add New** -> **Project**.
3. Chon repository GitHub cua du an.
4. O man hinh cau hinh, giu mac dinh vi repo da co `vercel.json`.
5. Framework Preset co the de **Other**.
6. Root Directory de la thu muc goc chua `app.py`.
7. Bam **Deploy**.

### 3. Them Environment Variables

Trong Vercel Dashboard:

1. Vao project vua tao.
2. Chon **Settings** -> **Environment Variables**.
3. Them cac bien can dung, toi thieu nen co:

```text
ADMIN_PASSWORD=<mat-khau-admin-cua-ban>
ADMIN_PATH=<duong-dan-admin-bi-mat>
INFERENCE_API_URL=<url-api-inference-cua-ban>
```

Neu dung Football-Data API:

```text
FOOTBALL_API_TOKEN=<token-football-data>
```

Neu dung Vercel Blob cho photobooth frame upload:

```text
BLOB_READ_WRITE_TOKEN=<token-vercel-blob>
```

Sau khi them bien moi truong, vao tab **Deployments** va bam **Redeploy** deployment moi nhat.

## Deploy bang Vercel CLI

### 1. Cai Vercel CLI

```bash
npm install -g vercel
```

### 2. Dang nhap

```bash
vercel login
```

### 3. Link project

Chay lenh trong thu muc goc du an:

```bash
vercel
```

Lan dau CLI se hoi mot so cau hoi:

- Set up and deploy? Chon `Y`.
- Which scope? Chon tai khoan/team cua ban.
- Link to existing project? Chon `N` neu tao moi.
- Project name? Nhap ten du an.
- Directory? De mac dinh `./`.
- Override settings? Thuong chon `N`.

### 4. Them bien moi truong bang CLI

```bash
vercel env add ADMIN_PASSWORD
vercel env add ADMIN_PATH
vercel env add INFERENCE_API_URL
vercel env add FOOTBALL_API_TOKEN
vercel env add BLOB_READ_WRITE_TOKEN
```

Bien nao khong dung thi co the bo qua.

### 5. Deploy production

```bash
vercel --prod
```

Sau khi xong, CLI se in ra URL production.

## Cau hinh Vercel cua du an

File `vercel.json` hien tai:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    },
    {
      "src": "static/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

Y nghia:

- Moi request khong phai static se duoc route vao `app.py`.
- File trong `static/` duoc phuc vu nhu static asset.
- Vercel se cai dependencies tu `requirements.txt`.

## Kiem tra sau khi deploy

Sau khi co URL Vercel, test cac duong dan:

```text
https://<ten-du-an>.vercel.app/
https://<ten-du-an>.vercel.app/photobooth
https://<ten-du-an>.vercel.app/thuyettrinh
https://<ten-du-an>.vercel.app/api/danh-sach-doi
https://<ten-du-an>.vercel.app/api/thuat-toan
https://<ten-du-an>.vercel.app/api/du-doan-vong-bang
https://<ten-du-an>.vercel.app/api/du-doan-knockout
```

Neu co cau hinh `ADMIN_PATH`, trang admin nam tai:

```text
https://<ten-du-an>.vercel.app/<ADMIN_PATH>
```

## Loi thuong gap khi deploy Vercel

### 1. Trang bi loi 500

Vao **Vercel Dashboard** -> project -> **Logs** de xem loi. Nguyen nhan hay gap:

- Thieu file CSV trong `data_processed`.
- Thieu file CSV trong `outputs`.
- Sai hoac thieu bien `INFERENCE_API_URL`.
- Endpoint inference tu xa dang sleep hoac loi.

### 2. Anh/bieu do khong hien thi

Kiem tra:

- Thu muc `static/anh` da duoc commit len GitHub chua.
- Duong dan anh trong app co dung `/static/...` khong.
- File anh co bi ignore trong `.gitignore` hoac `.vercelignore` khong.

### 3. Du doan dong bi loi model

Neu log bao thieu file trong `models/`, nghia la app dang fallback sang model local. Tren Vercel, `models/` dang bi ignore. Hay:

- Kiem tra `INFERENCE_API_URL`.
- Dam bao API inference tu xa dang chay.
- Neu API can token, them `INFERENCE_API_TOKEN`.

### 4. Lich thi dau chinh thuc khong cap nhat

API `/api/lich-thi-dau-chinh-thuc` can `FOOTBALL_API_TOKEN`. Neu khong co token hoac bi rate limit, app co the khong lay duoc du lieu moi tu Football-Data.

### 5. Upload/xoa khung photobooth khong hoat dong tren Vercel

Vercel Serverless khong phu hop de ghi file truc tiep lau dai vao filesystem. App co ho tro `BLOB_READ_WRITE_TOKEN`; hay cau hinh Vercel Blob neu muon luu khung photobooth sau deploy.

## Bao mat

- Doi `ADMIN_PASSWORD` truoc khi public project.
- Doi `ADMIN_PATH` thanh duong dan kho doan.
- Khong commit file `.env`, token Vercel, token Football-Data hoac API token.
- Kiem tra `.gitignore` truoc khi push.

## Ghi chu ve Docker va Hugging Face

Repo co `Dockerfile` va thu muc `hf_space/` cho cach deploy khac ngoai Vercel. Neu deploy Hugging Face Space, xem them `hf_space/README.md`.

## License va muc dich su dung

Du an phuc vu muc dich hoc tap, nghien cuu va minh hoa ung dung Machine Learning trong phan tich bong da. Ket qua du doan chi mang tinh tham khao, khong phai khuyen nghi ca cuoc.
