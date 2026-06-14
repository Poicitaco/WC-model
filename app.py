import os
import json
import urllib.error
import urllib.request
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Thư mục gốc chứa dữ liệu
DUONG_DAN_GOC = os.path.dirname(os.path.abspath(__file__))

# Tải các tệp dữ liệu processed
duong_dan_lich_su = os.path.join(DUONG_DAN_GOC, "data_processed", "matches_clean_final.csv")
duong_dan_fifa = os.path.join(DUONG_DAN_GOC, "data_processed", "fifa_rank_clean.csv")
duong_dan_elo = os.path.join(DUONG_DAN_GOC, "data_processed", "elo_current_clean.csv")
duong_dan_wc_exp = os.path.join(DUONG_DAN_GOC, "data_processed", "world_cup_experience_by_team.csv")
duong_dan_wc_groups = os.path.join(DUONG_DAN_GOC, "data_processed", "wc2026_groups_clean.csv")
duong_dan_mo_hinh = os.path.join(DUONG_DAN_GOC, "models", "best_model_final.pkl")
duong_dan_khung_anh = os.path.join(DUONG_DAN_GOC, "static", "photobooth_frames")
duong_dan_cau_hinh_khung = os.path.join(duong_dan_khung_anh, "frames.json")
MAT_KHAU_ADMIN = os.environ.get("ADMIN_PASSWORD", "24022941")
DUONG_DAN_ADMIN_AN = os.environ.get("ADMIN_PATH", "quan-tri-khung-anh-24022941")
DINH_DANG_KHUNG_CHO_PHEP = {".png", ".webp", ".svg"}
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "https://itentad-wc-2026-model-inference.hf.space").rstrip("/")
INFERENCE_API_TOKEN = os.environ.get("INFERENCE_API_TOKEN", "")
FOOTBALL_API_TOKEN = os.environ.get("FOOTBALL_API_TOKEN", "")
# Dữ liệu kết quả đã mô phỏng sẵn trong thư mục outputs
duong_dan_kq_vong_bang = os.path.join(DUONG_DAN_GOC, "outputs", "group_stage_predictions_with_scores.csv")
duong_dan_bxh_vong_bang = os.path.join(DUONG_DAN_GOC, "outputs", "group_standings_best_model.csv")
duong_dan_kq_knockout = os.path.join(DUONG_DAN_GOC, "outputs", "knockout_predictions_best_model.csv")

BO_NHO_LICH_API = {"du_lieu": None, "thoi_gian": 0}


def lay_lich_thi_dau_tu_api():
    if not FOOTBALL_API_TOKEN:
        return None

    # URL API Football-Data.org (World Cup: WC)
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    headers = {"X-Auth-Token": FOOTBALL_API_TOKEN}
    yeu_cau = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(yeu_cau, timeout=15) as phan_hoi:
            # Kiểm tra headers để theo dõi giới hạn (Rate Limiting)
            gioi_han_con_lai = phan_hoi.getheader("X-Requests-Available-Minute")
            if gioi_han_con_lai:
                print(f"API Football - Số lượt gọi còn lại trong phút: {gioi_han_con_lai}")

            du_lieu = json.loads(phan_hoi.read().decode("utf-8"))
            danh_sach_tran = []
            from datetime import datetime, timedelta
            for tran in du_lieu.get("matches", []):
                doi_a = tran["homeTeam"]["name"]
                doi_b = tran["awayTeam"]["name"]

                score_obj = tran.get("score", {})
                
                def get_score(period):
                    p = score_obj.get(period)
                    return (p.get("home"), p.get("away")) if p else (None, None)

                h_ft, a_ft = get_score("fullTime")
                h_rt, a_rt = get_score("regularTime")
                h_ht, a_ht = get_score("halfTime")

                bt_a = h_ft if h_ft is not None else (h_rt if h_rt is not None else h_ht)
                bt_b = a_ft if a_ft is not None else (a_rt if a_rt is not None else a_ht)

                # Chuyển đổi từ UTC sang Giờ Việt Nam (GMT+7)
                utc_date_str = tran.get("utcDate", "").replace("Z", "+00:00")
                utc_dt = datetime.fromisoformat(utc_date_str)
                vn_dt = utc_dt + timedelta(hours=7)
                ngay_gio_vn = vn_dt.strftime("%Y-%m-%d %H:%M")

                # Dịch tên vòng đấu sang Tiếng Việt
                vong_goc = str(tran.get("stage", "")).upper()
                vong_vn = vong_goc.replace("_", " ").title()
                if vong_goc == "GROUP_STAGE": vong_vn = "Vòng bảng"
                elif vong_goc == "LAST_32": vong_vn = "Vòng 1/16"
                elif vong_goc == "LAST_16": vong_vn = "Vòng 1/8"
                elif vong_goc == "QUARTER_FINALS": vong_vn = "Tứ kết"
                elif vong_goc == "SEMI_FINALS": vong_vn = "Bán kết"
                elif vong_goc == "FINAL": vong_vn = "Chung kết"
                elif vong_goc == "THIRD_PLACE": vong_vn = "Tranh hạng ba"

                danh_sach_tran.append({
                    "vong": vong_vn,
                    "ma_tran": tran.get("id"),
                    "bang": str(tran.get("group", "")).replace("GROUP_", ""),
                    "ngay_gio": ngay_gio_vn,
                    "doi_a": chuan_hoa_ten_doi(doi_a),
                    "doi_b": chuan_hoa_ten_doi(doi_b),
                    "ban_thang_a": int(bt_a) if bt_a is not None else 0,
                    "ban_thang_b": int(bt_b) if bt_b is not None else 0,
                    "trang_thai": tran.get("status")
                })
            return danh_sach_tran
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("API Football - Đang bị giới hạn lượt gọi (Rate Limit). Đợi 1 phút.")
        else:
            print(f"Lỗi HTTP từ Football API: {e.code}")
        return None
    except Exception as e:
        print(f"Lỗi hệ thống khi gọi Football API: {e}")
        return None


# Đọc các bảng dữ liệu
bo_du_lieu_tran_dau = pd.read_csv(duong_dan_lich_su, encoding="utf-8-sig")
bo_du_lieu_fifa = pd.read_csv(duong_dan_fifa, encoding="utf-8-sig")
bo_du_lieu_elo = pd.read_csv(duong_dan_elo, encoding="utf-8-sig")
bo_du_lieu_wc_exp = pd.read_csv(duong_dan_wc_exp, encoding="utf-8-sig")
bo_du_lieu_wc_groups = pd.read_csv(duong_dan_wc_groups, encoding="utf-8-sig")

# Chuẩn hóa cột ngày
bo_du_lieu_tran_dau["date"] = pd.to_datetime(bo_du_lieu_tran_dau["date"], errors="coerce")
bo_du_lieu_fifa["date"] = pd.to_datetime(bo_du_lieu_fifa["date"], errors="coerce")

# Chuẩn hóa bàn thắng bàn thua
for cot in ["ban_thang_a", "ban_thang_b"]:
    if cot in bo_du_lieu_tran_dau.columns:
        bo_du_lieu_tran_dau[cot] = pd.to_numeric(bo_du_lieu_tran_dau[cot], errors="coerce").fillna(0)

# Tải mô hình đã được huấn luyện tốt nhất
class MoHinhHeuristic:
    classes_ = np.array([0, 1, 2])

    def __init__(self, che_do):
        self.che_do = che_do

    def predict_proba(self, X_match):
        dong = X_match.iloc[0]
        p_elo = float(np.clip(dong["elo_expected_win_prob"], 0.02, 0.98))
        if self.che_do == "hybrid":
            diem_bo_sung = (
                float(dong["chenh_lech_fifa_rank"]) / 160
                + float(dong["chenh_lech_phong_do"]) / 9
                + float(dong["chenh_lech_kinh_nghiem_wc"]) / 30
            )
            logit = np.log(p_elo / (1 - p_elo)) + diem_bo_sung
            p_elo = 1 / (1 + np.exp(-logit))

        p_hoa = float(np.clip(0.38 - abs(p_elo - 0.5) * 0.5, 0.16, 0.38))
        phan_con_lai = 1 - p_hoa
        p_a_thang = p_elo * phan_con_lai
        p_a_thua = (1 - p_elo) * phan_con_lai
        return np.array([[p_a_thua, p_hoa, p_a_thang]])

    def predict(self, X_match):
        return np.array([int(np.argmax(self.predict_proba(X_match)[0]))])


CAU_HINH_THUAT_TOAN = {
    "random_forest_tuned": {
        "ten": "Random Forest Tuned",
        "mo_ta": "Mô hình mặc định, cân bằng tốt giữa độ chính xác và khả năng tổng quát.",
        "duong_dan": duong_dan_mo_hinh
    },
    "random_forest_baseline": {
        "ten": "Random Forest Baseline",
        "mo_ta": "Mô hình rừng ngẫu nhiên gốc để so sánh.",
        "duong_dan": os.path.join(DUONG_DAN_GOC, "models", "random_forest_baseline.pkl")
    },
    "elo_rating": {
        "ten": "ELO Rating",
        "mo_ta": "Thuật toán nhẹ dựa trên chênh lệch sức mạnh ELO giữa hai đội.",
        "loai": "heuristic",
        "che_do": "elo"
    },
    "elo_fifa_hybrid": {
        "ten": "ELO + FIFA Hybrid",
        "mo_ta": "Kết hợp ELO, FIFA Rank, phong độ và kinh nghiệm World Cup.",
        "loai": "heuristic",
        "che_do": "hybrid"
    },
    "xgboost_baseline": {
        "ten": "XGBoost Baseline",
        "mo_ta": "Boosting nhanh, có hiệu quả tốt trên tập kiểm thử.",
        "duong_dan": os.path.join(DUONG_DAN_GOC, "models", "xgboost_baseline.pkl")
    },
    "xgboost_tuned": {
        "ten": "XGBoost Tuned",
        "mo_ta": "XGBoost đã tinh chỉnh tham số.",
        "duong_dan": os.path.join(DUONG_DAN_GOC, "models", "xgboost_tuned.pkl")
    }
}
BO_NHO_MO_HINH = {}


def tai_mo_hinh(ten_thuat_toan):
    if ten_thuat_toan not in CAU_HINH_THUAT_TOAN:
        raise ValueError("Thuật toán không hợp lệ")
    if ten_thuat_toan not in BO_NHO_MO_HINH:
        cau_hinh = CAU_HINH_THUAT_TOAN[ten_thuat_toan]
        if cau_hinh.get("loai") == "heuristic":
            BO_NHO_MO_HINH[ten_thuat_toan] = MoHinhHeuristic(cau_hinh["che_do"])
            return BO_NHO_MO_HINH[ten_thuat_toan]
        duong_dan = cau_hinh["duong_dan"]
        if not os.path.exists(duong_dan):
            raise FileNotFoundError("Model chưa được triển khai trên máy chủ")
        BO_NHO_MO_HINH[ten_thuat_toan] = joblib.load(duong_dan)
    return BO_NHO_MO_HINH[ten_thuat_toan]


def goi_inference_tu_xa(ten_thuat_toan, X_match):
    if not INFERENCE_API_URL:
        raise RuntimeError("Chưa cấu hình inference API từ xa")

    du_lieu_gui = json.dumps({
        "model": ten_thuat_toan,
        "features": {cot: float(X_match.iloc[0][cot]) for cot in danh_sach_cot_dac_trung}
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if INFERENCE_API_TOKEN:
        headers["Authorization"] = f"Bearer {INFERENCE_API_TOKEN}"

    yeu_cau = urllib.request.Request(
        f"{INFERENCE_API_URL}/predict",
        data=du_lieu_gui,
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(yeu_cau, timeout=90) as phan_hoi:
            return json.loads(phan_hoi.read().decode("utf-8"))
    except urllib.error.HTTPError as loi:
        chi_tiet = loi.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference API trả lỗi {loi.code}: {chi_tiet}") from loi
    except urllib.error.URLError as loi:
        raise RuntimeError("Không thể kết nối máy chủ model từ xa") from loi

# Danh sách 12 cột đặc trưng đầu vào cho mô hình
danh_sach_cot_dac_trung = [
    "chenh_lech_phong_do",
    "chenh_lech_ban_thang_tb",
    "chenh_lech_ban_thua_tb",
    "chenh_lech_doi_dau",
    "chenh_lech_fifa_rank",
    "chenh_lech_elo",
    "elo_expected_win_prob",
    "chenh_lech_kinh_nghiem_wc",
    "chenh_lech_thang_wc",
    "san_trung_lap",
    "is_host_team_A",
    "is_host_team_B"
]

# Bản đồ ánh xạ chuẩn hóa tên các quốc gia
ANH_XA_TEN_DOI = {
    "USA": "United States",
    "United States of America": "United States",
    "USMNT": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "DR Congo": "Democratic Republic of the Congo",
    "Congo DR": "Democratic Republic of the Congo",
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
    "Macedonia": "North Macedonia"
}

def chuan_hoa_ten_doi(ten_doi_goc):
    if pd.isna(ten_doi_goc):
        return ten_doi_goc
    ten_chuan = str(ten_doi_goc).strip()
    return ANH_XA_TEN_DOI.get(ten_chuan, ten_chuan)


def doc_danh_sach_khung_anh():
    danh_sach = []
    if os.path.exists(duong_dan_cau_hinh_khung):
        with open(duong_dan_cau_hinh_khung, "r", encoding="utf-8") as tep:
            danh_sach = json.load(tep)

    # Đọc thêm từ Vercel Blob nếu có
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if token:
        try:
            req = urllib.request.Request("https://blob.vercel-storage.com/")
            req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=5) as res:
                blobs = json.loads(res.read()).get("blobs", [])
                for b in blobs:
                    pn = b.get("pathname", "")
                    if pn.startswith("frames/"):
                        ten_file = pn[7:]
                        cuts = 2
                        cac_phan = ten_file.split("__", 2)
                        if len(cac_phan) == 3 and cac_phan[0] in {"2cut", "4cut"}:
                            cuts = int(cac_phan[0][0])
                            ten_hien_thi, ten_goc = cac_phan[1], cac_phan[2]
                        elif "__" in ten_file:
                            ten_hien_thi, ten_goc = ten_file.split("__", 1)
                        else:
                            ten_hien_thi = os.path.splitext(ten_file)[0]
                            ten_goc = ten_file

                        danh_sach.append({
                            "id": b.get("url"),
                            "ten": ten_hien_thi.replace("_", " "),
                            "tep": ten_goc,
                            "url": b.get("url"),
                            "cuts": cuts
                        })
        except Exception as e:
            print("Lỗi đọc Vercel Blob:", e)

    return danh_sach


def luu_danh_sach_khung_anh(danh_sach):
    # Cập nhật mảng local
    danh_sach_local = [k for k in danh_sach if not str(k.get("url", "")).startswith("http")]
    os.makedirs(duong_dan_khung_anh, exist_ok=True)
    with open(duong_dan_cau_hinh_khung, "w", encoding="utf-8") as tep:
        json.dump(danh_sach_local, tep, ensure_ascii=False, indent=2)


def admin_hop_le():
    mat_khau = request.headers.get("X-Admin-Password") or request.form.get("mat_khau", "")
    return mat_khau == MAT_KHAU_ADMIN

# Áp dụng chuẩn hóa cho các bảng dữ liệu khi khởi chạy
for cot in ["doi_a", "doi_b"]:
    if cot in bo_du_lieu_tran_dau.columns:
        bo_du_lieu_tran_dau[cot] = bo_du_lieu_tran_dau[cot].apply(chuan_hoa_ten_doi)

if "team" in bo_du_lieu_fifa.columns:
    bo_du_lieu_fifa["team"] = bo_du_lieu_fifa["team"].apply(chuan_hoa_ten_doi)

if "team" in bo_du_lieu_elo.columns:
    bo_du_lieu_elo["team"] = bo_du_lieu_elo["team"].apply(chuan_hoa_ten_doi)

if "team" in bo_du_lieu_wc_exp.columns:
    bo_du_lieu_wc_exp["team"] = bo_du_lieu_wc_exp["team"].apply(chuan_hoa_ten_doi)

if "doi" in bo_du_lieu_wc_groups.columns:
    bo_du_lieu_wc_groups["doi"] = bo_du_lieu_wc_groups["doi"].apply(chuan_hoa_ten_doi)


def lay_thong_ke_phong_do_doi_bong(doi, ngay_dau, so_tran=5):
    doi = chuan_hoa_ten_doi(doi)
    ngay_dau = pd.to_datetime(ngay_dau)

    lich_su = bo_du_lieu_tran_dau[
        ((bo_du_lieu_tran_dau["doi_a"] == doi) | (bo_du_lieu_tran_dau["doi_b"] == doi)) &
        (bo_du_lieu_tran_dau["date"] < ngay_dau)
    ].sort_values("date").tail(so_tran)

    if lich_su.empty:
        return {
            "phong_do_5": 0.0,
            "ban_thang_tb_5": 0.0,
            "ban_thua_tb_5": 0.0
        }

    diem_phong_do = []
    so_ban_thang = []
    so_ban_thua = []

    for _, dong in lich_su.iterrows():
        if dong["doi_a"] == doi:
            bt = float(dong["ban_thang_a"])
            bb = float(dong["ban_thang_b"])
        else:
            bt = float(dong["ban_thang_b"])
            bb = float(dong["ban_thang_a"])

        if bt > bb:
            diem_phong_do.append(3)
        elif bt == bb:
            diem_phong_do.append(1)
        else:
            diem_phong_do.append(0)

        so_ban_thang.append(bt)
        so_ban_thua.append(bb)

    return {
        "phong_do_5": float(np.mean(diem_phong_do)),
        "ban_thang_tb_5": float(np.mean(so_ban_thang)),
        "ban_thua_tb_5": float(np.mean(so_ban_thua))
    }


def lay_diem_doi_dau(doi_a, doi_b, ngay_dau, so_tran=5):
    doi_a = chuan_hoa_ten_doi(doi_a)
    doi_b = chuan_hoa_ten_doi(doi_b)
    ngay_dau = pd.to_datetime(ngay_dau)

    doi_dau = bo_du_lieu_tran_dau[
        (((bo_du_lieu_tran_dau["doi_a"] == doi_a) & (bo_du_lieu_tran_dau["doi_b"] == doi_b)) |
         ((bo_du_lieu_tran_dau["doi_a"] == doi_b) & (bo_du_lieu_tran_dau["doi_b"] == doi_a))) &
        (bo_du_lieu_tran_dau["date"] < ngay_dau)
    ].sort_values("date").tail(so_tran)

    if doi_dau.empty:
        return 0.0

    diem_doi_dau = []
    for _, dong in doi_dau.iterrows():
        if dong["doi_a"] == doi_a:
            bt_a = float(dong["ban_thang_a"])
            bt_b = float(dong["ban_thang_b"])
        else:
            bt_a = float(dong["ban_thang_b"])
            bt_b = float(dong["ban_thang_a"])

        if bt_a > bt_b:
            diem_doi_dau.append(3)
        elif bt_a == bt_b:
            diem_doi_dau.append(1)
        else:
            diem_doi_dau.append(0)

    return float(np.mean(diem_doi_dau))


def lay_hang_fifa_moi_nhat(doi, ngay_dau):
    doi = chuan_hoa_ten_doi(doi)
    ngay_dau = pd.to_datetime(ngay_dau)

    if "team" not in bo_du_lieu_fifa.columns:
        return 210.0

    cot_hang = None
    for ten_cot in ["rank", "ranking", "fifa_rank"]:
        if ten_cot in bo_du_lieu_fifa.columns:
            cot_hang = ten_cot
            break

    if cot_hang is None:
        return 210.0

    lich_su_hang = bo_du_lieu_fifa[
        (bo_du_lieu_fifa["team"] == doi) &
        (bo_du_lieu_fifa["date"] <= ngay_dau)
    ].sort_values("date")

    if lich_su_hang.empty:
        lich_su_tat_ca = bo_du_lieu_fifa[bo_du_lieu_fifa["team"] == doi].sort_values("date")
        if lich_su_tat_ca.empty:
            return 210.0
        return float(lich_su_tat_ca.iloc[-1][cot_hang])

    return float(lich_su_hang.iloc[-1][cot_hang])


def lay_elo_hien_tai(doi):
    doi = chuan_hoa_ten_doi(doi)
    if "team" not in bo_du_lieu_elo.columns:
        return 1500.0

    thong_tin_elo = bo_du_lieu_elo[bo_du_lieu_elo["team"] == doi]
    if thong_tin_elo.empty:
        return 1500.0

    for ten_cot in ["elo_rating", "rating", "elo"]:
        if ten_cot in thong_tin_elo.columns:
            return float(thong_tin_elo.iloc[0][ten_cot])

    return 1500.0


def tinh_elo_mong_doi(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def lay_kinh_nghiem_wc(doi):
    doi = chuan_hoa_ten_doi(doi)
    if "team" not in bo_du_lieu_wc_exp.columns:
        return {
            "so_lan_du_wc": 0.0,
            "so_tran_wc": 0.0,
            "so_tran_thang_wc": 0.0
        }

    thong_tin_kinh_nghiem = bo_du_lieu_wc_exp[bo_du_lieu_wc_exp["team"] == doi]
    if thong_tin_kinh_nghiem.empty:
        return {
            "so_lan_du_wc": 0.0,
            "so_tran_wc": 0.0,
            "so_tran_thang_wc": 0.0
        }

    dong_tin = thong_tin_kinh_nghiem.iloc[0]
    return {
        "so_lan_du_wc": float(dong_tin.get("so_lan_du_wc", dong_tin.get("wc_appearances", 0))),
        "so_tran_wc": float(dong_tin.get("so_tran_wc", dong_tin.get("wc_matches", 0))),
        "so_tran_thang_wc": float(dong_tin.get("so_tran_thang_wc", dong_tin.get("wc_wins", 0)))
    }


def la_doi_chu_nha(doi):
    doi = chuan_hoa_ten_doi(doi)
    return 1 if doi in ["United States", "Mexico", "Canada"] else 0


def tao_dac_trung_tran_dau(doi_a, doi_b, ngay_dau, san_trung_lap=1):
    doi_a = chuan_hoa_ten_doi(doi_a)
    doi_b = chuan_hoa_ten_doi(doi_b)
    ngay_dau = pd.to_datetime(ngay_dau)

    phong_do_a = lay_thong_ke_phong_do_doi_bong(doi_a, ngay_dau)
    phong_do_b = lay_thong_ke_phong_do_doi_bong(doi_b, ngay_dau)

    doi_dau_a = lay_diem_doi_dau(doi_a, doi_b, ngay_dau)
    doi_dau_b = lay_diem_doi_dau(doi_b, doi_a, ngay_dau)

    hang_a = lay_hang_fifa_moi_nhat(doi_a, ngay_dau)
    hang_b = lay_hang_fifa_moi_nhat(doi_b, ngay_dau)

    elo_a = lay_elo_hien_tai(doi_a)
    elo_b = lay_elo_hien_tai(doi_b)

    wc_a = lay_kinh_nghiem_wc(doi_a)
    wc_b = lay_kinh_nghiem_wc(doi_b)

    dong_dac_trung = {
        "chenh_lech_phong_do": phong_do_a["phong_do_5"] - phong_do_b["phong_do_5"],
        "chenh_lech_ban_thang_tb": phong_do_a["ban_thang_tb_5"] - phong_do_b["ban_thang_tb_5"],
        "chenh_lech_ban_thua_tb": phong_do_a["ban_thua_tb_5"] - phong_do_b["ban_thua_tb_5"],
        "chenh_lech_doi_dau": doi_dau_a - doi_dau_b,
        "chenh_lech_fifa_rank": hang_b - hang_a,  # Rank nhỏ hơn tốt hơn
        "chenh_lech_elo": elo_a - elo_b,
        "elo_expected_win_prob": tinh_elo_mong_doi(elo_a, elo_b),
        "chenh_lech_kinh_nghiem_wc": wc_a["so_lan_du_wc"] - wc_b["so_lan_du_wc"],
        "chenh_lech_thang_wc": wc_a["so_tran_thang_wc"] - wc_b["so_tran_thang_wc"],
        "san_trung_lap": int(san_trung_lap),
        "is_host_team_A": la_doi_chu_nha(doi_a),
        "is_host_team_B": la_doi_chu_nha(doi_b)
    }

    df_dac_trung = pd.DataFrame([dong_dac_trung])
    df_dac_trung = df_dac_trung[danh_sach_cot_dac_trung].copy()
    return df_dac_trung, dong_dac_trung


def lay_xac_suat_ket_qua(model, X_match):
    xac_suat_tho = model.predict_proba(X_match)[0]
    cac_lop = list(model.classes_)

    ti_le = {
        "p_a_thua": 0.0,
        "p_hoa": 0.0,
        "p_a_thang": 0.0
    }

    for lop, p in zip(cac_lop, xac_suat_tho):
        if lop == 0:
            ti_le["p_a_thua"] = float(p)
        elif lop == 1:
            ti_le["p_hoa"] = float(p)
        elif lop == 2:
            ti_le["p_a_thang"] = float(p)

    return ti_le


def giai_ma_nhan_du_doan(nhan):
    if nhan == 0:
        return "Đội A thua"
    elif nhan == 1:
        return "Hòa"
    elif nhan == 2:
        return "Đội A thắng"
    return "Không xác định"


def gia_lap_ty_so_tran_dau(nhan, ti_le_dict):
    p_a_thua = ti_le_dict["p_a_thua"]
    p_hoa = ti_le_dict["p_hoa"]
    p_a_thang = ti_le_dict["p_a_thang"]

    if nhan == 2:  # Đội A thắng
        if p_a_thang >= 0.75:
            return 3, 1
        elif p_a_thang >= 0.60:
            return 2, 0
        else:
            return 2, 1
    elif nhan == 1:  # Hòa
        if p_hoa >= 0.65:
            return 1, 1
        elif p_hoa >= 0.50:
            return 2, 2
        else:
            return 0, 0
    elif nhan == 0:  # Đội A thua (Đội B thắng)
        if p_a_thua >= 0.75:
            return 1, 3
        elif p_a_thua >= 0.60:
            return 0, 2
        else:
            return 1, 2
    return 0, 0


# Router chính trả về giao diện Dashboard
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/photobooth")
def trang_photobooth():
    return render_template("photobooth.html")


@app.route("/thuyettrinh")
def thuyet_trinh():
    from flask import redirect
    return redirect("/static/presentation/index.html")


@app.route(f"/{DUONG_DAN_ADMIN_AN}")
def trang_admin_khung_anh():
    return render_template("admin_photobooth.html")


@app.route("/api/khung-anh", methods=["GET"])
def api_khung_anh():
    return jsonify(doc_danh_sach_khung_anh())


@app.route("/api/admin/khung-anh", methods=["POST"])
def api_them_khung_anh():
    if not admin_hop_le():
        return jsonify({"loi": "Sai mật khẩu quản trị"}), 401

    tep = request.files.get("khung_anh")
    ten_hien_thi = request.form.get("ten_hien_thi", "").strip()
    so_cut = request.form.get("so_cut", "2").strip()
    if so_cut not in {"2", "4"}:
        return jsonify({"loi": "Số cut chỉ có thể là 2 hoặc 4"}), 400
    if not tep or not tep.filename:
        return jsonify({"loi": "Chưa chọn tệp khung ảnh"}), 400

    ten_tep = secure_filename(tep.filename)
    phan_mo_rong = os.path.splitext(ten_tep)[1].lower()
    if phan_mo_rong not in DINH_DANG_KHUNG_CHO_PHEP:
        return jsonify({"loi": "Chỉ hỗ trợ PNG, WEBP hoặc SVG"}), 400

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if token:
        # Nếu có token thì upload lên Vercel Blob
        try:
            ten_hien_thi_blob = ten_hien_thi.replace(" ", "_") if ten_hien_thi else os.path.splitext(ten_tep)[0].replace("-", "_")
            pathname = f"frames/{so_cut}cut__{ten_hien_thi_blob}__{ten_tep}"
            url = f"https://blob.vercel-storage.com/{urllib.parse.quote(pathname)}"

            req = urllib.request.Request(url, data=tep.read(), method="PUT")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", tep.content_type)
            with urllib.request.urlopen(req) as res:
                blob_info = json.loads(res.read())

            return jsonify({"thanh_cong": True, "khung_anh": {
                "id": blob_info.get("url"),
                "ten": ten_hien_thi or os.path.splitext(ten_tep)[0].replace("-", " ").title(),
                "tep": ten_tep,
                "url": blob_info.get("url"),
                "cuts": int(so_cut)
            }})
        except Exception as e:
            return jsonify({"loi": f"Lỗi upload Vercel Blob: {str(e)}"}), 503
    else:
        # Fallback lưu vào ổ đĩa khi chạy local không có token
        try:
            os.makedirs(duong_dan_khung_anh, exist_ok=True)
            tep.save(os.path.join(duong_dan_khung_anh, ten_tep))
            danh_sach = [khung for khung in doc_danh_sach_khung_anh() if khung["tep"] != ten_tep]
            danh_sach.append({
                "id": os.path.splitext(ten_tep)[0],
                "ten": ten_hien_thi or os.path.splitext(ten_tep)[0].replace("-", " ").title(),
                "tep": ten_tep,
                "url": f"/static/photobooth_frames/{ten_tep}",
                "cuts": int(so_cut)
            })
            luu_danh_sach_khung_anh(danh_sach)
            return jsonify({"thanh_cong": True, "khung_anh": danh_sach[-1]})
        except OSError:
            return jsonify({
                "loi": f"Máy chủ hiện không cho phép ghi tệp. Token hiện tại: {'Chưa có (None)' if not token else 'Đã có nhưng lỗi'}. Vui lòng kiểm tra biến môi trường BLOB_READ_WRITE_TOKEN."
            }), 503


@app.route("/api/admin/khung-anh", methods=["DELETE"])
def api_xoa_khung_anh():
    khung_id = request.json.get("id") if request.is_json else None
    if not khung_id:
        return jsonify({"loi": "Thiếu ID khung ảnh"}), 400
    if not admin_hop_le():
        return jsonify({"loi": "Sai mật khẩu quản trị"}), 401

    danh_sach = doc_danh_sach_khung_anh()
    khung_can_xoa = next((khung for khung in danh_sach if khung["id"] == khung_id), None)
    if not khung_can_xoa:
        return jsonify({"loi": "Không tìm thấy khung ảnh"}), 404

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if khung_id.startswith("http") and token:
        try:
            req = urllib.request.Request("https://blob.vercel-storage.com/delete", method="POST")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            data = json.dumps({"urls": [khung_id]}).encode("utf-8")
            with urllib.request.urlopen(req, data=data) as res:
                pass
            return jsonify({"thanh_cong": True})
        except Exception as e:
            return jsonify({"loi": f"Lỗi xóa Vercel Blob: {str(e)}"}), 503
    else:
        try:
            duong_dan_tep = os.path.join(duong_dan_khung_anh, khung_can_xoa["tep"])
            if os.path.exists(duong_dan_tep):
                os.remove(duong_dan_tep)
            luu_danh_sach_khung_anh([khung for khung in danh_sach if khung["id"] != khung_id])
            return jsonify({"thanh_cong": True})
        except OSError:
            return jsonify({"loi": "Máy chủ hiện không cho phép xóa tệp"}), 503


# API 1: Trả về danh sách tất cả các đội tuyển và bảng đấu
@app.route("/api/danh-sach-doi", methods=["GET"])
def api_danh_sach_doi():
    danh_sach_doi = []
    for _, dong in bo_du_lieu_wc_groups.iterrows():
        ten_doi = dong["doi"]
        chuan_ten = chuan_hoa_ten_doi(ten_doi)
        bang = dong["bang"]

        hang_fifa = lay_hang_fifa_moi_nhat(chuan_ten, "2026-06-01")
        elo = lay_elo_hien_tai(chuan_ten)
        wc_exp = lay_kinh_nghiem_wc(chuan_ten)

        danh_sach_doi.append({
            "ten": chuan_ten,
            "bang": bang,
            "fifa_rank": hang_fifa,
            "elo": elo,
            "so_lan_du_wc": wc_exp["so_lan_du_wc"],
            "so_tran_wc": wc_exp["so_tran_wc"],
            "so_tran_thang_wc": wc_exp["so_tran_thang_wc"],
            "la_chu_nha": la_doi_chu_nha(chuan_ten)
        })

    # Sắp xếp theo bảng đấu
    danh_sach_doi.sort(key=lambda x: (x["bang"], x["ten"]))
    return jsonify(danh_sach_doi)


@app.route("/api/thuat-toan", methods=["GET"])
def api_thuat_toan():
    danh_sach = []
    for ma, cau_hinh in CAU_HINH_THUAT_TOAN.items():
        la_model_tu_xa = bool(INFERENCE_API_URL) and cau_hinh.get("loai") != "heuristic"
        kha_dung = la_model_tu_xa or cau_hinh.get("loai") == "heuristic" or os.path.exists(cau_hinh.get("duong_dan", ""))
        ly_do = ""
        if kha_dung and ma.startswith("xgboost") and not la_model_tu_xa:
            try:
                import xgboost  # noqa: F401
            except ImportError:
                kha_dung = False
                ly_do = "Máy chủ chưa cài thư viện XGBoost"
        elif not kha_dung:
            ly_do = "Model chưa được triển khai"

        danh_sach.append({
            "ma": ma,
            "ten": cau_hinh["ten"],
            "mo_ta": cau_hinh["mo_ta"],
            "kha_dung": kha_dung,
            "ly_do": ly_do,
            "nguon": "Hugging Face inference" if la_model_tu_xa else "Máy chủ hiện tại"
        })
    return jsonify(danh_sach)


# API 2: Nhận yêu cầu dự đoán trận đấu động
@app.route("/api/du-doan-tran-dau", methods=["POST"])
def api_du_doan_tran_dau():
    du_lieu_nhan = request.json
    if not du_lieu_nhan:
        return jsonify({"loi": "Du lieu yeu cau khong hop le!"}), 400

    doi_a = du_lieu_nhan.get("doi_a")
    doi_b = du_lieu_nhan.get("doi_b")
    ngay_dau = du_lieu_nhan.get("ngay_gio", "2026-06-11")
    san_trung_lap = du_lieu_nhan.get("san_trung_lap", 1)
    ten_thuat_toan = du_lieu_nhan.get("thuat_toan", "random_forest_tuned")

    if not doi_a or not doi_b:
        return jsonify({"loi": "Thieu ten doi bong!"}), 400

    if doi_a == doi_b:
        return jsonify({"loi": "Hai doi bong phai khac nhau!"}), 400

    try:
        X_tran_dau, chi_tiet_dac_trung = tao_dac_trung_tran_dau(
            doi_a=doi_a,
            doi_b=doi_b,
            ngay_dau=ngay_dau,
            san_trung_lap=san_trung_lap
        )

        cau_hinh_thuat_toan = CAU_HINH_THUAT_TOAN.get(ten_thuat_toan)
        if not cau_hinh_thuat_toan:
            return jsonify({"loi": "Thuật toán không hợp lệ"}), 400

        if INFERENCE_API_URL and cau_hinh_thuat_toan.get("loai") != "heuristic":
            try:
                ket_qua_tu_xa = goi_inference_tu_xa(ten_thuat_toan, X_tran_dau)
                nhan_du_doan = int(ket_qua_tu_xa["nhan_du_doan"])
                xac_suat_ti_le = ket_qua_tu_xa["ti_le"]
                nguon_inference = "Hugging Face inference"
            except Exception as e:
                print(f"Fallback to local model due to remote API error: {e}")
                try:
                    mo_hinh_da_chon = tai_mo_hinh(ten_thuat_toan)
                    nhan_du_doan = int(mo_hinh_da_chon.predict(X_tran_dau)[0])
                    xac_suat_ti_le = lay_xac_suat_ket_qua(mo_hinh_da_chon, X_tran_dau)
                    nguon_inference = "Máy chủ hiện tại (Fallback)"
                except FileNotFoundError:
                    return jsonify({"loi": f"Hugging Face API lỗi ({e}) và máy chủ Vercel không có model nội bộ."}), 500
        else:
            mo_hinh_da_chon = tai_mo_hinh(ten_thuat_toan)
            nhan_du_doan = int(mo_hinh_da_chon.predict(X_tran_dau)[0])
            xac_suat_ti_le = lay_xac_suat_ket_qua(mo_hinh_da_chon, X_tran_dau)
            nguon_inference = "Máy chủ hiện tại"

        ket_qua_du_doan = giai_ma_nhan_du_doan(nhan_du_doan)

        ban_thang_a, ban_thang_b = gia_lap_ty_so_tran_dau(nhan_du_doan, xac_suat_ti_le)

        doi_thang = doi_a if nhan_du_doan == 2 else (doi_b if nhan_du_doan == 0 else "Hòa")

        # Thống kê bổ sung cho giao diện
        thong_tin_doi_a = {
            "ten": doi_a,
            "fifa_rank": lay_hang_fifa_moi_nhat(doi_a, ngay_dau),
            "elo": lay_elo_hien_tai(doi_a),
            "wc_exp": lay_kinh_nghiem_wc(doi_a),
            "phong_do": lay_thong_ke_phong_do_doi_bong(doi_a, ngay_dau)
        }
        thong_tin_doi_b = {
            "ten": doi_b,
            "fifa_rank": lay_hang_fifa_moi_nhat(doi_b, ngay_dau),
            "elo": lay_elo_hien_tai(doi_b),
            "wc_exp": lay_kinh_nghiem_wc(doi_b),
            "phong_do": lay_thong_ke_phong_do_doi_bong(doi_b, ngay_dau)
        }

        ket_qua_tra_ve = {
            "doi_a": doi_a,
            "doi_b": doi_b,
            "thuat_toan": ten_thuat_toan,
            "ten_thuat_toan": CAU_HINH_THUAT_TOAN[ten_thuat_toan]["ten"],
            "nguon_inference": nguon_inference,
            "nhan_du_doan": nhan_du_doan,
            "ket_qua_du_doan": ket_qua_du_doan,
            "doi_thang": doi_thang,
            "ban_thang_a": ban_thang_a,
            "ban_thang_b": ban_thang_b,
            "ty_so": f"{ban_thang_a} - {ban_thang_b}",
            "ti_le": xac_suat_ti_le,
            "dac_trung": chi_tiet_dac_trung,
            "thong_tin_doi_a": thong_tin_doi_a,
            "thong_tin_doi_b": thong_tin_doi_b
        }
        return jsonify(ket_qua_tra_ve)

    except Exception as e:
        return jsonify({"loi": f"Co loi xay ra: {str(e)}"}), 500


# API 3: Trả về kết quả vòng bảng đã mô phỏng sẵn
@app.route("/api/du-doan-vong-bang", methods=["GET"])
def api_du_doan_vong_bang():
    try:
        df_du_doan = pd.read_csv(duong_dan_kq_vong_bang, encoding="utf-8-sig").fillna("")
        df_bxh = pd.read_csv(duong_dan_bxh_vong_bang, encoding="utf-8-sig").fillna("")

        danh_sach_tran = df_du_doan.to_dict(orient="records")
        danh_sach_bxh = df_bxh.to_dict(orient="records")

        # Gom nhóm BXH theo bảng đấu
        bxh_theo_bang = {}
        for dong in danh_sach_bxh:
            bang_dau = dong["bang"]
            if bang_dau not in bxh_theo_bang:
                bxh_theo_bang[bang_dau] = []
            bxh_theo_bang[bang_dau].append(dong)

        # Sắp xếp mỗi bảng theo thứ hạng
        for bang_dau in bxh_theo_bang:
            bxh_theo_bang[bang_dau].sort(key=lambda x: x["hang"])

        return jsonify({
            "danh_sach_tran": danh_sach_tran,
            "bxh_theo_bang": bxh_theo_bang
        })
    except Exception as e:
        return jsonify({"loi": f"Khong the doc du lieu vong bang: {str(e)}"}), 500


# API 4: Trả về kết quả vòng loại trực tiếp knockout đã mô phỏng sẵn
@app.route("/api/du-doan-knockout", methods=["GET"])
def api_du_doan_knockout():
    try:
        df_knockout = pd.read_csv(duong_dan_kq_knockout, encoding="utf-8-sig").fillna("")
        danh_sach_tran_ko = df_knockout.to_dict(orient="records")
        return jsonify(danh_sach_tran_ko)
    except Exception as e:
        return jsonify({"loi": f"Khong the doc du lieu knockout: {str(e)}"}), 500


# API 5: Trả về lịch thi đấu đầy đủ, đã chuẩn hóa để hiển thị theo ngày
@app.route("/api/lich-thi-dau", methods=["GET"])
def api_lich_thi_dau():
    try:
        df_vong_bang = pd.read_csv(duong_dan_kq_vong_bang, encoding="utf-8-sig").fillna("")
        df_knockout = pd.read_csv(duong_dan_kq_knockout, encoding="utf-8-sig").fillna("")
        danh_sach_tran = []

        for _, tran in df_vong_bang.iterrows():
            danh_sach_tran.append({
                "vong": tran["vong"],
                "ma_tran": int(tran["ma_tran"]),
                "bang": tran["bang"],
                "ngay_gio": tran["ngay_gio"],
                "doi_a": tran["doi_a"],
                "doi_b": tran["doi_b"],
                "doi_thang": tran["doi_du_doan_thang"],
                "ban_thang_a": int(tran["ban_thang_a"]),
                "ban_thang_b": int(tran["ban_thang_b"]),
                "p_a_thang": float(tran["p_a_thang"]),
                "p_hoa": float(tran["p_hoa"]),
                "p_a_thua": float(tran["p_a_thua"])
            })

        for _, tran in df_knockout.iterrows():
            danh_sach_tran.append({
                "vong": tran["vong"],
                "ma_tran": int(tran["ma_tran"]),
                "bang": "",
                "ngay_gio": tran["ngay_gio"],
                "doi_a": tran["doi_a"],
                "doi_b": tran["doi_b"],
                "doi_thang": tran["doi_thang"],
                "ban_thang_a": int(tran["ban_thang_a"]),
                "ban_thang_b": int(tran["ban_thang_b"]),
                "p_a_thang": float(tran["p_a_thang"]),
                "p_hoa": float(tran["p_hoa"]),
                "p_a_thua": float(tran["p_a_thua"])
            })

        danh_sach_tran.sort(key=lambda tran: tran["ngay_gio"])
        cac_ngay = sorted({tran["ngay_gio"][:10] for tran in danh_sach_tran})

        return jsonify({
            "cac_ngay": cac_ngay,
            "danh_sach_tran": danh_sach_tran,
            "tong_so_tran": len(danh_sach_tran)
        })
    except Exception as e:
        return jsonify({"loi": f"Khong the doc lich thi dau: {str(e)}"}), 500


# API 6: Trả về số liệu đánh giá mô hình
@app.route("/api/lich-thi-dau-chinh-thuc", methods=["GET"])
def api_lich_thi_dau_chinh_thuc():
    import time
    bay_gio = time.time()

    # Thử lấy từ cache API trước
    danh_sach_tran = None
    if BO_NHO_LICH_API["du_lieu"] and (bay_gio - BO_NHO_LICH_API["thoi_gian"] < 600):
        danh_sach_tran = BO_NHO_LICH_API["du_lieu"]
    else:
        du_lieu_api = lay_lich_thi_dau_tu_api()
        if du_lieu_api:
            BO_NHO_LICH_API["du_lieu"] = du_lieu_api
            BO_NHO_LICH_API["thoi_gian"] = bay_gio
            danh_sach_tran = du_lieu_api

    if not danh_sach_tran:
        try:
            cac_tep = [
                os.path.join(DUONG_DAN_GOC, "data_processed", "wc2026_schedule_group_clean.csv"),
                os.path.join(DUONG_DAN_GOC, "data_processed", "wc2026_schedule_knockout_clean.csv")
            ]
            danh_sach_tran = []
            for duong_dan in cac_tep:
                if not os.path.exists(duong_dan): continue
                for _, tran in pd.read_csv(duong_dan, encoding="utf-8-sig").fillna("").iterrows():
                    danh_sach_tran.append({
                        "vong": tran["vong"],
                        "ma_tran": int(tran["ma_tran"]),
                        "bang": "" if str(tran["bang"]).lower() == "nan" else tran["bang"],
                        "ngay_gio": tran["ngay_gio"],
                        "doi_a": tran["doi_a"],
                        "doi_b": tran["doi_b"],
                        "ban_thang_a": 0,
                        "ban_thang_b": 0,
                        "trang_thai": "TIMED"
                    })
        except Exception as e:
            return jsonify({"loi": f"Khong thể đọc lịch thi đấu: {str(e)}"}), 500

    danh_sach_tran.sort(key=lambda tran: tran["ngay_gio"])
    return jsonify({
        "cac_ngay": sorted({tran["ngay_gio"][:10] for tran in danh_sach_tran}),
        "danh_sach_tran": danh_sach_tran,
        "tong_so_tran": len(danh_sach_tran)
    })


@app.route("/api/so-lieu-mo-hinh", methods=["GET"])
def api_so_lieu_mo_hinh():
    # Cung cấp tên hình ảnh phân tích đã được lưu
    cac_anh = {
        "bubble_chart": "/static/anh/bubble_chart_mo_phong_4_mo_hinh.png",
        "confusion_matrix_tuned": "/static/anh/confusion_matrix_random_forest_tuned.png",
        "feature_importance": "/static/anh/feature_importance_best_model.png",
        "accuracy_hat_graph": "/static/anh/hat_graph_train_test_accuracy_green_yellow.png",
        "model_radar": "/static/anh/model_comparison_radar_chart.png",
        "metrics_heatmap": "/static/anh/model_metrics_heatmap_soft_purple_yellow.png",
        "bump_chart": "/static/anh/model_ranking_bump_chart.png",
        "podium_compare": "/static/anh/top4_podium_mirror_rf_vs_xgb.png"
    }

    # Số liệu chi tiết của Random Forest Tuned so với các mô hình khác
    so_lieu_so_sanh = [
        {
            "mo_hinh": "Random Forest Baseline",
            "train_acc": 0.9997,
            "test_acc": 0.6407,
            "macro_f1": 0.5131,
            "weighted_f1": 0.5996,
            "overfit_gap": 0.3589
        },
        {
            "mo_hinh": "Random Forest Tuned",
            "train_acc": 0.7630,
            "test_acc": 0.6234,
            "macro_f1": 0.5577,
            "weighted_f1": 0.6223,
            "overfit_gap": 0.1396,
            "duoc_chon": True
        },
        {
            "mo_hinh": "XGBoost Baseline",
            "train_acc": 0.6757,
            "test_acc": 0.6600,
            "macro_f1": 0.5039,
            "weighted_f1": 0.5962,
            "overfit_gap": 0.0157
        },
        {
            "mo_hinh": "XGBoost Tuned",
            "train_acc": 0.6936,
            "test_acc": 0.5993,
            "macro_f1": 0.5503,
            "weighted_f1": 0.6065,
            "overfit_gap": 0.0943
        }
    ]

    return jsonify({
        "hinh_anh": cac_anh,
        "so_lieu": so_lieu_so_sanh
    })


@app.route("/api/bieu-do-ky-thuat", methods=["GET"])
def api_bieu_do_ky_thuat():
    danh_sach_bieu_do = [
        {
            "ten": "Phân Bố Mô Phỏng Bóng Bóng",
            "url": "/static/anh/bubble_chart_mo_phong_4_mo_hinh.png",
            "giai_thich": "Biểu đồ bong bóng so sánh mô phỏng 4 mô hình khác nhau, thể hiện độ lệch và hiệu suất phân loại của từng mô hình qua các kịch bản trận đấu."
        },
        {
            "ten": "Ma Trận Nhầm Lẫn Random Forest Baseline",
            "url": "/static/anh/confusion_matrix_random_forest_baseline.png",
            "giai_thich": "Ma trận nhầm lẫn của mô hình Random Forest Baseline, cho thấy mức độ dự đoán chính xác và các sai sót (nhầm lẫn) giữa các kết quả Thắng, Hòa, Thua."
        },
        {
            "ten": "Ma Trận Nhầm Lẫn Random Forest Tuned",
            "url": "/static/anh/confusion_matrix_random_forest_tuned.png",
            "giai_thich": "Ma trận nhầm lẫn của Random Forest sau khi đã tinh chỉnh siêu tham số, giúp cải thiện độ chính xác và giảm thiểu dự đoán sai."
        },
        {
            "ten": "Ma Trận Nhầm Lẫn XGBoost Baseline",
            "url": "/static/anh/confusion_matrix_xgboost_baseline.png",
            "giai_thich": "Ma trận nhầm lẫn cơ bản của mô hình XGBoost, cho thấy xu hướng dự đoán ban đầu chưa qua tối ưu."
        },
        {
            "ten": "Ma Trận Nhầm Lẫn XGBoost Tuned",
            "url": "/static/anh/confusion_matrix_xgboost_tuned.png",
            "giai_thich": "Ma trận nhầm lẫn của XGBoost sau khi được tối ưu hóa siêu tham số để cân bằng khả năng dự đoán."
        },
        {
            "ten": "Mức Độ Quan Trọng Của Đặc Trưng (Feature Importance)",
            "url": "/static/anh/feature_importance_best_model.png",
            "giai_thich": "Đánh giá mức độ đóng góp của từng yếu tố đầu vào (chênh lệch Elo, tỷ lệ thắng theo Elo, v.v.) vào kết quả quyết định của mô hình tốt nhất."
        },
        {
            "ten": "Độ Chính Xác Giữa Train & Test",
            "url": "/static/anh/hat_graph_train_test_accuracy_green_yellow.png",
            "giai_thich": "Biểu đồ mũ so sánh độ chính xác giữa tập dữ liệu huấn luyện và tập kiểm thử, giúp phát hiện tình trạng quá khớp (overfitting)."
        },
        {
            "ten": "Bảng Dự Đoán Vòng Loại Trực Tiếp (Chi Tiết)",
            "url": "/static/anh/knockout_predictions_table_best_model.png",
            "giai_thich": "Bảng dự đoán chi tiết tất cả các trận đấu vòng loại trực tiếp từ vòng 32 đội cho đến trận chung kết của mô hình tốt nhất."
        },
        {
            "ten": "Bảng Vòng Loại Trực Tiếp (Thu Gọn)",
            "url": "/static/anh/knockout_table_beautiful_compact.png",
            "giai_thich": "Bảng tổng hợp dự đoán vòng loại trực tiếp với giao diện thu gọn, thiết kế đẹp mắt và dễ nhìn."
        },
        {
            "ten": "Bảng Vòng Loại Trực Tiếp (Tối Giản)",
            "url": "/static/anh/knockout_table_clean_style.png",
            "giai_thich": "Bảng hiển thị các trận đấu vòng loại trực tiếp theo phong cách thiết kế tối giản, tập trung vào số liệu cốt lõi."
        },
        {
            "ten": "Bảng Vòng Loại Trực Tiếp (Đầy Đủ Cột)",
            "url": "/static/anh/knockout_table_full_columns_best_model.png",
            "giai_thich": "Bảng dự đoán vòng loại trực tiếp với đầy đủ tất cả các cột thống kê về xác suất, tỷ số và quyết định cuối cùng."
        },
        {
            "ten": "Bảng Vòng Loại Trực Tiếp (Không Quyết Định)",
            "url": "/static/anh/knockout_table_without_decision_best_model.png",
            "giai_thich": "Bảng thông tin dự đoán vòng loại trực tiếp tập trung vào các xác suất thắng/hòa/thua mà chưa bao gồm cột quyết định thắng cuộc."
        },
        {
            "ten": "Biểu Đồ Cột 3D Đánh Giá Mô Hình",
            "url": "/static/anh/model_3d_bar_bright.png",
            "giai_thich": "Biểu đồ cột 3D sáng màu so sánh trực quan hiệu năng và các điểm số đánh giá giữa nhiều mô hình."
        },
        {
            "ten": "Biểu Đồ Cột Nhóm So Sánh Hiệu Năng",
            "url": "/static/anh/model_comparison_grouped_bar.png",
            "giai_thich": "Biểu đồ cột nhóm so sánh chi tiết các chỉ số như Accuracy, Macro F1, Weighted F1 giữa các mô hình huấn luyện."
        },
        {
            "ten": "Biểu Đồ Radar Đánh Giá Đa Chiều",
            "url": "/static/anh/model_comparison_radar_chart.png",
            "giai_thich": "Biểu đồ Radar giúp đánh giá và so sánh toàn diện nhiều mô hình trên cùng một trục đa chiều (Accuracy, F1, Log Loss)."
        },
        {
            "ten": "Bản Đồ Nhiệt Chỉ Số (Heatmap)",
            "url": "/static/anh/model_metrics_heatmap_soft_purple_yellow.png",
            "giai_thich": "Bản đồ nhiệt (Heatmap) thể hiện tổng quan các chỉ số đánh giá của các mô hình, màu đậm hơn biểu thị hiệu năng cao hơn."
        },
        {
            "ten": "Biểu Đồ Xếp Hạng Biến Động",
            "url": "/static/anh/model_ranking_bump_chart.png",
            "giai_thich": "Biểu đồ xếp hạng (Bump chart) theo dõi sự thay đổi thứ hạng của các mô hình dựa trên nhiều loại thang đo chỉ số khác nhau."
        },
        {
            "ten": "Thẻ Chiến Đấu Top 4: RF vs XGBoost",
            "url": "/static/anh/top4_battle_cards_rf_vs_xgb.png",
            "giai_thich": "Giao diện thẻ chiến đấu so sánh mô phỏng top 4 đội mạnh nhất tiến sâu vào vòng trong giữa Random Forest và XGBoost."
        },
        {
            "ten": "Mức Độ Trùng Khớp Top 4",
            "url": "/static/anh/top4_overlap_rf_vs_xgb.png",
            "giai_thich": "Biểu đồ Venn hoặc đồ thị phần trăm thể hiện mức độ trùng khớp trong việc dự đoán 4 đội mạnh nhất giữa 2 mô hình."
        },
        {
            "ten": "Bục Vinh Quang Top 4",
            "url": "/static/anh/top4_podium_mirror_rf_vs_xgb.png",
            "giai_thich": "Hình ảnh đối xứng bục vinh quang so sánh thứ hạng của các đội tuyển đạt thành tích cao nhất theo dự đoán của RF và XGBoost."
        },
        {
            "ten": "Bảng Tham Khảo Knockout XGBoost",
            "url": "/static/anh/xgb_knockout_table_reference.png",
            "giai_thich": "Bảng tham khảo kết quả chi tiết các trận đấu vòng loại trực tiếp dựa trên dự đoán mô phỏng của XGBoost."
        }
    ]
    return jsonify(danh_sach_bieu_do)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
