# Text-to-SQL tiếng Việt: fine-tune theo vòng lặp milestone

## Vòng lặp
```
 MỘT LẦN, NGOÀI VÒNG LẶP:  bakeoff (so ứng viên trên val) ──► bạn ghi quyết định vào params.yaml: train.base_model
                                                                     │
          ┌──────────── vòng sau: đổi data / params, milestone.id mới ┼──────────────────┐
          ▼                                                          ▼                  │
fetch → prepare → check_data → sanity → train → eval_val → select → register → golden ─┘
 (DVC lineage)     (CPU)     (overfit 50) (grid×seed) (bảng val) (ổn định+ngưỡng) (MLflow) (1 lần/milestone)
```

| Tập | Dùng để | Nguồn |
|---|---|---|
| **train** (13 662) | fine-tune | ViText2SQL train (tiếng Việt) + câu tiếng Anh gốc của Spider trên cùng DB |
| **val** (952) | chọn base model (bakeoff), hyperparam, checkpoint, **ngưỡng** | ViText2SQL dev |
| **golden** (1 907) | chấm **1 lần ở cuối mỗi milestone**, quyết định promote | ViText2SQL test |

Ba tập không dùng chung database nào. SQL lấy từ Spider gốc (tiếng Anh, chạy được) vì ViText2SQL dịch cả tên
bảng/cột nên SQL của nó không chạy được (chi tiết trong `src/prepare.py`).

## Các bước và cổng chặn
1. **check_data** (CPU): 3 tập không chung DB; không rò rỉ câu hỏi; bộ chấm cho 100% khi dùng SQL vàng làm dự đoán
   và bắt được dự đoán sai; logic chọn ngưỡng đúng trên dữ liệu tổng hợp. Fail thì dừng.
2. **sanity** (GPU, vài phút): kiểm tra label masking (chỉ phần SQL được tính loss), rồi cho model học thuộc 50 mẫu
   và chấm lại chính 50 mẫu đó. **EX < 98% hoặc loss không xuống → pipeline có bug**, dừng trước khi tốn GPU.
3. **bakeoff (ngoài vòng lặp, file `bakeoff/dvc.yaml`)**: chạy một lần lúc đầu, hoặc khi muốn xét lại base model.
   Các ứng viên chạy lần lượt với cùng ngân sách (zero-shot, fine-tune ngắn, chấm val, đo latency, đếm tham số).
   Kết quả nằm ở `reports/bakeoff.md` cùng một đề xuất (trong ràng buộc `max_params_b`; model nhỏ hơn thắng nếu kém
   ≤ 2 điểm). **Bạn tự ghi quyết định** vào `train.base_model`. Không stage nào trong vòng lặp phụ thuộc vào bakeoff,
   nên sửa code hay dữ liệu không kéo nó chạy lại.
4. **train**: grid `lr × seed`; mỗi tổ hợp là một MLflow run, lưu checkpoint mỗi epoch.
5. **eval_val**: chấm mọi checkpoint trên val và trên `train_probe` (300 câu train) để có train-EX.
6. **select**: tạo `reports/val_table.md` gồm bảng từng checkpoint, cột chẩn đoán và bảng độ ổn định theo seed.
   - Chọn cấu hình có `mean − k·std` của val-EX cao nhất, ưu tiên cấu hình không có dấu hiệu under/overfit.
   - Ngưỡng confidence: lấy coverage lớn nhất mà precision trên val ≥ `target_precision`.
     Khi chạy thật: SQL chạy được **và** confidence ≥ ngưỡng thì trả lời, ngược lại từ chối.
7. **register**: đóng gói model + ngưỡng + code prompt thành MLflow pyfunc, đăng ký với alias `candidate`.
   Version được gắn tag val-EX, ngưỡng, md5 dữ liệu, git sha.
8. **golden**: chấm đúng artifact đã đăng ký. Đạt thì gắn alias `champion` (phải vượt champion cũ).
   Ghi `reports/golden/ledger.csv` và `reports/next_action.md` (gợi ý cho vòng sau).
   Cùng milestone mà model khác thì **bị từ chối**: phải mở milestone mới.

## Đọc chẩn đoán
| Hiện tượng | Kết luận | Làm gì |
|---|---|---|
| sanity < 98% | bug pipeline | sửa template / masking / eos / bộ chấm |
| train-EX thấp | underfit | tăng epoch hoặc lr, hoặc dùng model lớn hơn |
| train-EX cao, val-EX kém xa, val-loss tăng | overfit | giảm epoch hoặc lr, thêm dữ liệu |
| val-EX dao động mạnh giữa các seed | không ổn định | giảm lr, thêm seed |
| golden-EX ≪ val-EX | chọn quá khớp val / lệch phân phối | **không chỉnh theo golden**; mở rộng val |
| EX ổn nhưng precision@ngưỡng trên golden ≪ target | ngưỡng chọn trên val không chuyển giao | calibrate lại trên val, milestone mới |

## Luồng chạy

```
 LAPTOP (Windows)                         GOOGLE DRIVE / GITHUB              MÁY GPU THUÊ (Linux, RTX 4090)
 ─────────────────                        ─────────────────────              ──────────────────────────────
 B0  dvc repro check_data  ── dvc push ──►  Drive (dữ liệu, private)
                           ── git push ──►  GitHub (code, dvc.lock)  ──clone─► B1  cài môi trường
                                                                     ──pull──► B2  lấy dữ liệu, kiểm tra md5
                                                                               B3  dvc repro sanity    (cổng chặn)
                                                                               B4  dvc repro           (cả vòng)
 B6  đọc kết quả, quyết định  ◄── git pull / scp ◄─────────────────────────── B5  lưu kết quả TRƯỚC khi tắt máy
 B7  milestone mới → quay lại B3/B4
```

### B0. Laptop: chuẩn bị (đã xong)
```bash
.venv\Scripts\python -m dvc repro check_data   # tải dữ liệu, tạo train/val/golden, kiểm tra trên CPU
.venv\Scripts\python -m dvc push               # đẩy dữ liệu lên Drive (lần đầu mở trình duyệt để đăng nhập)
git push -u origin main                        # code + params.yaml + dvc.yaml + dvc.lock
```
Secret OAuth của Drive nằm trong `.env` và `.dvc/config.local` (đều không lên git).
Token đăng nhập Drive được lưu ở `%LOCALAPPDATA%\pydrive2fs\Cache\<client_id>\default.json`; máy GPU cần file này.

### B1. Máy GPU: cài môi trường
Thuê RTX 4090 24GB (vast.ai / RunPod, template PyTorch + CUDA), ổ đĩa ≥ 100GB vì checkpoint fp32 khá nặng.
```bash
git clone https://github.com/giangkh1908/Finetune_Qwen3.5-0.8B.git && cd Finetune_Qwen3.5-0.8B
pip install -r requirements.txt          # flash-linear-attention / causal-conv1d build lỗi thì bỏ, chỉ chậm hơn
tmux new -s ft                           # mọi lệnh dài chạy trong tmux, mất SSH không chết job
```

### B2. Máy GPU: lấy dữ liệu và kiểm tra lineage
Từ laptop, copy token Drive lên máy GPU:
```bash
scp "%LOCALAPPDATA%\pydrive2fs\Cache\<client_id>\default.json" root@<ip-gpu>:~/gdrive-creds.json
```
Trên máy GPU:
```bash
dvc remote modify --local storage gdrive_client_id <CLIENT_ID>          # lấy trong .env ở laptop
dvc remote modify --local storage gdrive_client_secret <CLIENT_SECRET>
dvc remote modify --local storage gdrive_user_credentials_file ~/gdrive-creds.json
dvc pull                       # kéo raw/ + data/ đúng theo md5 trong dvc.lock
dvc status                     # "Data and pipelines are up to date" = dữ liệu khớp từng byte với laptop
```
Không có Drive thì `dvc repro check_data` tự tải lại từ nguồn gốc, rồi `git diff dvc.lock` phải không có thay đổi.

### B3. Máy GPU: sanity (cổng chặn, vài phút)
```bash
dvc repro sanity
cat reports/sanity.json        # "pass": true mới được đi tiếp
```
Fail thì **dừng**, xem `reports/sanity_wrong.jsonl` (cột `pred` so với `sql`, cột `raw_output`):
loss không xuống → lỗi train (lr, label masking); loss thấp mà EX thấp → lỗi inference (template, eos) hoặc bộ chấm.

### B3.5. Máy GPU: bakeoff (chỉ lần đầu, khoảng 1–1.5 giờ)
```bash
dvc repro bakeoff/dvc.yaml
cat reports/bakeoff.md         # bảng so sánh + model được đề xuất
```
Ghi model đã chọn vào `params.yaml` → `train.base_model`. Nếu khác model đang dùng cho sanity thì chạy lại B3.
Đã chốt base model rồi thì các milestone sau bỏ qua bước này.

### B4. Máy GPU: chạy cả vòng (khoảng 6–10 giờ)
```bash
dvc repro 2>&1 | tee logs_M1.txt        # train → eval_val → select → register → golden
```
Theo dõi trong lúc chạy (cửa sổ tmux khác: `Ctrl+b c`):
```bash
nvidia-smi -l 5                                         # VRAM, OOM thì giảm train.batch_size và tăng grad_accum
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# trên laptop: ssh -L 5000:localhost:5000 root@<ip-gpu>, rồi mở http://localhost:5000
```
Máy bị ngắt giữa chừng: chạy lại `dvc repro`. Run đã xong (có file `DONE`) và dự đoán đã chấm được giữ lại.

### B5. Máy GPU: lưu kết quả TRƯỚC KHI TẮT MÁY
Tắt máy thuê là mất hết thứ chưa mang về.
```bash
git add dvc.lock bakeoff/ reports/ && git commit -m "M1: kết quả" && git push     # bảng val, bakeoff, golden, ledger
dvc push                                                                  # nếu có dữ liệu mới
tar czf mlflow_M1.tgz mlflow.db mlartifacts/                              # tracking + registry + model đã đăng ký
```
Từ laptop: `scp root@<ip-gpu>:~/Finetune_Qwen3.5-0.8B/mlflow_M1.tgz .`
(Hoặc từ đầu đặt `MLFLOW_TRACKING_URI` trỏ tới một server MLflow riêng, khi đó không cần bước này.)

### B6. Laptop: đọc kết quả, quyết định
```bash
git pull
```
Đọc theo thứ tự:
1. `reports/val_table.md`: từng checkpoint, chẩn đoán under/overfit, độ ổn định qua seed, checkpoint + ngưỡng được chọn.
2. `reports/next_action.md`: kết quả golden, có promote `champion` không, vòng sau nên làm gì.
3. `reports/golden/ledger.csv`: lịch sử golden qua các milestone.

### B7. Vòng sau (milestone mới)
Sửa đúng thứ mà `next_action.md` chỉ ra, đổi `milestone.id`, rồi quay lại B4 (đổi code thì chạy lại B3 trước):
```bash
dvc exp run -S milestone.id=M2 -S 'milestone.note=giảm epoch vì overfit' -S train.epochs=2
dvc exp show                   # so params + metrics giữa các vòng
```
DVC chỉ chạy lại các stage bị ảnh hưởng. Ví dụ đổi `train.*` (trừ `base_model`) thì không chạy lại fetch/prepare/sanity.
Golden của milestone cũ không chấm lại được; model mới bắt buộc thuộc milestone mới.

### Dùng model đã promote
```bash
python -m src.infer --db my.sqlite --question "Có bao nhiêu khách hàng ở Hà Nội?"   # alias champion
```
Mỗi lần gọi là một trace trong experiment `text2sql-inference` của MLflow (schema → generate → execute → quyết định).

## GPU
- sanity / train 0.8B: RTX 4090 24GB.
- bakeoff: 3 ứng viên ≤ 1B chạy lần lượt trên cùng 1 RTX 4090 (~1–1.5 giờ). Thêm model 1.5–2B thì đặt `method: lora` hoặc thuê A100 80GB.
- Ước tính cho grid mặc định (2 lr × 2 seed × 3 epoch) với 0.8B trên 4090: khoảng 6–10 giờ train + 1–2 giờ eval.
  Thu nhỏ grid nếu muốn rẻ hơn.

## License
ViText2SQL chỉ dùng cho nghiên cứu/giáo dục, **không phân phối lại**. Không public `data/`, model đã train hay DVC remote.
