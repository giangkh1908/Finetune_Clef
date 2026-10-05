# Clef-flash tiếng Việt: phân loại intent trên MASSIVE vi-VN, fine-tune theo vòng lặp milestone

**Mục tiêu:** đo xem Clef-flash ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9B, Apache-2.0)
xử lý tiếng Việt tốt đến đâu, và fine-tune thêm thì được bao nhiêu, **trước khi** đầu tư gán nhãn dữ liệu thật.
Dữ liệu: [MASSIVE](https://github.com/alexa/massive) vi-VN của Amazon (60 intent trợ lý ảo, người dịch, **CC-BY-4.0**,
dùng được cho thương mại).

Clef không sinh text: nhận `state` (câu của người dùng) + câu hỏi kiểu `choice` (60 intent kèm mô tả) và trả
**xác suất cho từng intent** trong một forward pass. Fine-tune bằng
[clef-finetune](https://github.com/MersivMedia/clef-finetune): LoRA trên backbone Qwen3.5-9B + train joint schema head,
loss CE làm mượt + Brier (công thức trong bài công bố của Cloudflare).

## Vòng lặp
```
 MỘT LẦN, NGOÀI VÒNG LẶP:  bakeoff (Clef-flash vs Laya, zero-shot + fine-tune 500/2000 câu) ──► bạn ghi train.base_model
                                                                     │
          ┌──────────── vòng sau: đổi data / schema / params, milestone.id mới ┼─────────────────┐
          ▼                                                          ▼                         │
fetch → prepare → check_data → sanity → train → eval_val → select → register → golden ────────┘
 (sha256)  (chia)    (CPU)    (overfit 50) (lr×seed) (bảng val) (ổn định+ngưỡng) (MLflow) (1 lần/milestone)
```

| Tập | Tỷ lệ | Số câu | Dùng để |
|---|---|---|---|
| **train** | 82% của 85% còn lại (≈69.6%) | 11 504 | fine-tune |
| **val** | 18% của 85% còn lại (≈15.4%) | 2 539 | bakeoff, hyperparam, checkpoint, **ngưỡng** |
| **golden** | 15% tổng | 2 478 | chấm **1 lần ở cuối mỗi milestone**, quyết định promote |

Gộp cả 16 521 câu của MASSIVE vi-VN rồi chia lại (`prepare.split`): phân tầng theo intent (intent nào cũng có ở cả
3 tập) và **câu trùng chữ luôn nằm cùng một tập**. Split gốc của MASSIVE có ~5% câu ngắn trùng y hệt giữa
train và test ("tăng âm lượng"), ở đây là 0%. Đổi lại, số liệu không so được 1-1 với số công bố trên split gốc
(mỗi câu vẫn giữ `orig_partition` để đối chiếu).

## Version: cái gì được pin, ở đâu
| Thứ | Pin bằng | Ở đâu |
|---|---|---|
| Dữ liệu nguồn MASSIVE 1.1 | sha256 của tarball (lệch thì `fetch` dừng) | `params.yaml: fetch` |
| raw/, data/ (train/val/golden) | md5 từng file | `dvc.lock` (git) + DVC remote (Drive) |
| Schema = mô tả 60 intent mà model đọc | md5 | `schema/` (git), deps trong `dvc.yaml`, tag MLflow `schema.*.md5` |
| Clef-flash, Clef, Laya | commit Hugging Face | `params.yaml: train.base_revision`, `bakeoff.candidates[].revision` |
| clef-finetune | git commit | `requirements-gpu.txt` |
| Model đã train | MLflow Model Registry version + alias `candidate`/`champion` | `mlflow.db` + `mlartifacts/` |
| Kết quả mỗi vòng | milestone id | `reports/golden/ledger.csv`, `history/` |

Mỗi MLflow run và model version mang tag md5 của dữ liệu + schema + `dvc.lock` + git sha: từ một model truy ngược được
đúng byte dữ liệu và code đã tạo ra nó. Quay lại version cũ: `git checkout <commit> && dvc checkout` (hoặc `dvc pull`).

## Các bước và cổng chặn
1. **check_data** (CPU): 3 tập không chung id, không có câu trùng chữ giữa train và val/golden; tỷ lệ chia đúng;
   intent nào cũng có ở val/golden; **phân phối intent/scenario của từng tập khớp phân phối chung** (TVD ≤
   `check.max_label_dist_tvd`, hiện train 0.0015 / val 0.0059 / golden 0.0039); mọi request hợp lệ theo **đúng validator của clef-finetune**; thứ tự option
   khớp encoder của Cloudflare (sắp xếp alphabet); bộ chấm cho 100% khi dự đoán = nhãn, 0% khi sai, NLL = log 60 khi
   phân bố đều; logic chọn ngưỡng đúng trên dữ liệu tổng hợp. Fail thì dừng.
2. **sanity** (GPU, vài phút): (a) prompt + schema 60 intent + câu dài nhất vừa `eval.max_length` (encoder của Cloudflare
   *lặng lẽ cắt* state nếu thiếu chỗ); (b) chấm theo batch cho cùng đáp án với từng câu; (c) LoRA học thuộc 50 câu rồi
   chấm lại: **acc < 98% hoặc loss không xuống → pipeline có bug**, dừng trước khi tốn GPU.
3. **bakeoff (ngoài vòng lặp, `bakeoff/dvc.yaml`)**: trả lời câu hỏi chính.
   - zero-shot trên val: Clef-flash, `laya-vi` (Laya 0.3B tinh chỉnh tiếng Việt), `laya-multilingual`;
     tuỳ chọn Clef 27B (chỉ zero-shot, cần GPU ≥ 80GB).
   - Clef-flash fine-tune ngắn với **500 và 2000 câu**: đường "gán nhãn thêm N câu thì được bao nhiêu điểm".
   - acc, macro-F1, ECE, **coverage tự động @ precision mục tiêu**, latency, số tham số → `reports/bakeoff.md`.
   Laya chỉ chấm zero-shot (pipeline chỉ fine-tune được Clef). **Bạn tự ghi quyết định** vào `train.base_model`.
4. **train**: grid `lr × seed` bằng clef-finetune; mỗi tổ hợp là một MLflow run, checkpoint (adapter + head) mỗi epoch.
5. **eval_val**: chấm mọi checkpoint trên val và trên `train_probe` (300 câu train) để có train-acc.
6. **select**: `reports/val_table.md` gồm bảng từng checkpoint, cột chẩn đoán, độ ổn định theo seed, intent yếu nhất,
   cặp nhầm nhiều nhất.
   - Chọn cấu hình có `mean − k·std` của val-acc cao nhất, ưu tiên cấu hình không under/overfit/miscalibrated.
   - Ngưỡng: quét mọi ngưỡng confidence trên val (`reports/threshold_curve.csv`, `dvc plots show`), lấy ngưỡng
     **F1 max trong các ngưỡng qua cổng chất lượng** (không ngưỡng nào qua thì F1 max toàn bộ, cổng FAIL).
     Khi chạy thật: confidence ≥ ngưỡng thì **tự động xử lý**, ngược lại **chuyển người** (hoặc hỏi lại người dùng).
7. **register**: đóng gói adapter + head + schema + ngưỡng thành MLflow pyfunc. **Mỗi lần chạy = 1 version mới**
   trong Registry (kể cả khi trượt cổng, để ghi nhận), mang tag TP/FP/FN/TN, P/R/F1/FPR trên val + `gate_val`.
   Chỉ version qua cổng val mới có alias `candidate`. Backbone 9B không nằm trong artifact: tải từ Hugging Face ở
   đúng revision đã pin.
8. **golden**: trượt cổng val thì không chấm (golden của milestone còn nguyên). Còn lại: chấm đúng artifact đã đăng ký
   tại ngưỡng của val → cổng golden + F1/acc không tụt quá xa val → version "eligible".
   **Champion = version eligible có golden F1 cao nhất trong mọi version** (cùng tập golden; hoà thì precision cao
   hơn, rồi version mới hơn). Ghi `reports/golden/ledger.csv` và `reports/next_action.md` (kèm bảng xếp hạng).
   Cùng milestone mà model khác thì **bị từ chối**.

### Cổng chất lượng (`params.yaml: gate`)
Bài toán nhị phân "tự động hay chuyển người" tại ngưỡng t, positive = model trả lời đúng:

| | confidence ≥ t (tự động) | confidence < t (chuyển người) |
|---|---|---|
| **model đúng** | TP | FN (chuyển người thừa) |
| **model sai** | FP (lỗi lọt ra ngoài) | TN (chặn được lỗi) |

Precision = TP/(TP+FP) · Recall = TP/(TP+FN) · F1 · FPR = FP/(FP+TN) · coverage = (TP+FP)/n.
Cổng: `min_precision`, `min_recall`, `min_f1`, `max_fpr`; áp trên val (register) và trên golden (champion).
Giá trị hiện tại là điểm khởi đầu, chỉnh sau M1 khi đã thấy đường cong thật.

## Đọc chẩn đoán
| Hiện tượng | Kết luận | Làm gì |
|---|---|---|
| sanity fail | bug pipeline | xem `reports/sanity.json` (pass_length / pass_batch / pass_loss / pass_acc) |
| train-acc thấp | underfit | tăng epoch, lr hoặc `lora.r` |
| train-acc cao, val-acc kém xa, val-NLL tăng | overfit | giảm epoch/lr, tăng augmentation, thêm dữ liệu |
| ECE cao | confidence không đáng tin | tăng `brier_weight` / `label_smoothing` |
| val-acc dao động mạnh giữa các seed | không ổn định | giảm lr, thêm seed |
| một cặp intent hay nhầm | mô tả trong schema chưa phân biệt | sửa `schema/massive_intent.yaml`, milestone mới |
| acc trên câu 3/3 người chấm đồng ý ≫ acc chung | nhãn MASSIVE mơ hồ | `prepare.min_agree_train`, gán nhãn lại |
| golden-acc ≪ val-acc | chọn quá khớp val | **không chỉnh theo golden**; mở rộng val |
| F1@ngưỡng trên golden ≪ trên val | ngưỡng không chuyển giao | calibrate lại trên val, milestone mới |
| cổng FAIL ở precision / FPR | câu sai vẫn tự tin cao | calibration (ECE), dữ liệu cho cặp hay nhầm |
| cổng FAIL ở recall | câu đúng bị đẩy sang người | confidence thấp: tăng epoch, giảm `label_smoothing` |

Biết trước: `cooking_query` chỉ có 6 câu trong toàn bộ MASSIVE vi-VN (4 ở train), intent này sẽ yếu.

## Luồng chạy

```
 LAPTOP (Windows)                         GOOGLE DRIVE / GITHUB              MÁY GPU THUÊ (Linux, A100/H100 80GB)
 ─────────────────                        ─────────────────────              ──────────────────────────────────
 B0  dvc repro check_data  ── dvc push ──►  Drive (dữ liệu, private)
                           ── git push ──►  GitHub (code, dvc.lock)  ──clone─► B1  cài môi trường
                                                                     ──pull──► B2  lấy dữ liệu, kiểm tra md5
                                                                               B3  dvc repro sanity    (cổng chặn)
                                                                               B3.5 bakeoff            (lần đầu)
                                                                               B4  dvc repro           (cả vòng)
 B6  đọc kết quả, quyết định  ◄── git pull / scp ◄─────────────────────────── B5  lưu kết quả TRƯỚC khi tắt máy
 B7  milestone mới → quay lại B3/B4
```

### B0. Laptop: chuẩn bị
```bash
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps "clef-finetune @ git+https://github.com/MersivMedia/clef-finetune@c83f7bc1acc23c81ad0de838510c59c48bf60c79"
.venv\Scripts\python -m dvc repro check_data   # tải MASSIVE, chia train/val/golden, kiểm tra trên CPU
.venv\Scripts\python -m dvc push               # đẩy dữ liệu lên Drive
git push                                       # code + params.yaml + schema/ + dvc.yaml + dvc.lock
```
Secret OAuth của Drive nằm trong `.env` và `.dvc/config.local` (đều không lên git).
Token đăng nhập Drive được lưu ở `%LOCALAPPDATA%\pydrive2fs\Cache\<client_id>\default.json`; máy GPU cần file này.

### B1. Máy GPU: cài môi trường
Clef-flash là 9B: LoRA cần **1 GPU 80GB** (A100/H100), theo hướng dẫn của clef-finetune. 4090 24GB không đủ cho
train (chỉ trọng số bf16 đã ~18GB); L40S/A6000 48GB có thể chạy nếu giảm `eval.max_length`, chưa kiểm chứng.
Ổ đĩa ≥ 100GB (backbone ~18GB + checkpoint).
```bash
git clone <repo> && cd <repo>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt      # flash-linear-attention / causal-conv1d build lỗi thì bỏ, chỉ chậm hơn
huggingface-cli download Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6   # tải trước
tmux new -s ft                           # mọi lệnh dài chạy trong tmux, mất SSH không chết job
```

### B2. Máy GPU: lấy dữ liệu và kiểm tra lineage
```bash
scp "%LOCALAPPDATA%\pydrive2fs\Cache\<client_id>\default.json" root@<ip-gpu>:~/gdrive-creds.json   # từ laptop
dvc remote modify --local storage gdrive_client_id <CLIENT_ID>          # lấy trong .env ở laptop
dvc remote modify --local storage gdrive_client_secret <CLIENT_SECRET>
dvc remote modify --local storage gdrive_user_credentials_file ~/gdrive-creds.json
dvc pull                       # kéo raw/ + data/ đúng theo md5 trong dvc.lock
dvc status                     # "Data and pipelines are up to date" = dữ liệu khớp từng byte với laptop
```
MASSIVE là dữ liệu public: không có Drive thì `dvc repro check_data` tự tải lại (sha256 được kiểm tra),
rồi `git diff dvc.lock` phải không có thay đổi.

### B3. Máy GPU: sanity (cổng chặn, ~15 phút kể cả tải model)
```bash
dvc repro sanity
cat reports/sanity.json        # "pass": true mới được đi tiếp
```

### B3.5. Máy GPU: bakeoff (chỉ lần đầu, ~1 giờ)
```bash
dvc repro bakeoff/dvc.yaml
cat reports/bakeoff.md         # bảng so sánh + đề xuất + đường fine-tune 500/2000 câu
```
Ghi model đã chọn vào `params.yaml` → `train.base_model` + `train.base_revision`. Khác model đang dùng thì chạy lại B3.

### B4. Máy GPU: chạy cả vòng
```bash
dvc repro 2>&1 | tee logs_M1.txt        # train → eval_val → select → register → golden
```
Ước tính grid mặc định (2 lr × 2 seed × 2 epoch, 11 504 câu, ~1–2k token/câu) trên H100: khoảng 1–1.5 giờ/epoch/run
→ 5–10 giờ train + ~1 giờ chấm. **Ước tính, chưa đo**: chạy `dvc repro sanity` trước và xem thời gian/step trong log.
Thu nhỏ grid (1 lr × 2 seed) nếu muốn rẻ hơn.
```bash
nvidia-smi -l 5
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # laptop: ssh -L 5000:localhost:5000 root@<ip-gpu>
```
Máy bị ngắt giữa chừng: chạy lại `dvc repro`. Run đã xong (có `DONE`) được giữ; run dở chạy tiếp từ checkpoint mới nhất.

### B5. Máy GPU: lưu kết quả TRƯỚC KHI TẮT MÁY
```bash
git add dvc.lock bakeoff/ reports/ && git commit -m "M1: kết quả" && git push
tar czf mlflow_M1.tgz mlflow.db mlartifacts/            # tracking + registry + adapter/head đã đăng ký (vài trăm MB)
```
Từ laptop: `scp root@<ip-gpu>:~/<repo>/mlflow_M1.tgz .`

### B6. Laptop: đọc kết quả, quyết định
1. `reports/bakeoff.md` (lần đầu): Clef-flash tiếng Việt zero-shot ra sao, fine-tune thêm 500/2000 câu được bao nhiêu.
2. `reports/val_table.md`: từng checkpoint, chẩn đoán, độ ổn định qua seed, intent yếu, ngưỡng được chọn.
3. `reports/next_action.md`: golden, có promote `champion` không, vòng sau nên làm gì.
4. `reports/golden/ledger.csv`: lịch sử golden qua các milestone.

### B7. Vòng sau (milestone mới)
```bash
dvc exp run -S milestone.id=M2 -S 'milestone.note=sửa mô tả intent hay nhầm' -S train.epochs=3
dvc exp show
```
Sửa `schema/massive_intent.yaml` (mô tả intent) cũng là một thay đổi hợp lệ cho milestone mới: DVC chạy lại từ sanity.

### Dùng model đã promote
```bash
python -m src.infer --text "đặt báo thức lúc sáu giờ sáng mai"     # alias champion
```
Ra intent, confidence, top 3, và `decision` = `auto` (≥ ngưỡng) hoặc `human`. Mỗi lần gọi là một trace trong
experiment `intent-inference` của MLflow. Cần thư mục release độc lập (merge LoRA, load bằng code của Cloudflare /
vLLM): `clef-finetune merge --model Cloudflare/clef-flash --checkpoint <checkpoint> --output releases/<tên>`.

## Lưu ý
- clef-finetune còn alpha: tác giả mới kiểm thử trên CPU với model tí hon, **chưa có lần train GPU nào trên trọng số
  thật**. Sanity ở đây là lần kiểm chứng đầu tiên, nên đọc kỹ `reports/sanity.json`.
- clef-finetune đọc file bằng encoding mặc định của hệ điều hành: chạy train trên Linux (UTF-8). Trên Windows đặt
  `PYTHONUTF8=1`.

## License
MASSIVE: CC-BY-4.0 (ghi nguồn: FitzGerald et al., 2022, *MASSIVE: A 1M-Example Multilingual Natural Language
Understanding Dataset*). Clef / Clef-flash, clef-finetune, Laya: Apache-2.0.
