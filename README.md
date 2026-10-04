# Text-to-SQL tiếng Việt: fine-tune theo vòng lặp milestone

## Vòng lặp
```
          ┌──────────────── vòng sau: đổi data / params / model, milestone.id mới ───────────────┐
          ▼                                                                                       │
fetch → prepare → check_data → sanity → bakeoff → train → eval_val → select → register → golden ─┘
 (DVC lineage)     (CPU)       (overfit 50)  (chọn model)  (grid×seed)  (bảng val)  (ổn định+ngưỡng)  (MLflow)  (1 lần/milestone)
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
3. **bakeoff**: mọi ứng viên trong `params.yaml` chạy cùng ngân sách (zero-shot, fine-tune ngắn, chấm val, đo latency,
   đếm tham số). Kết quả ở `reports/bakeoff.md`. Chọn trong ràng buộc `max_params_b`; model nhỏ hơn thắng nếu kém ≤ 2 điểm.
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

## Chạy
```bash
pip install -r requirements.txt

dvc repro check_data          # laptop (CPU) cũng chạy được
dvc repro sanity              # GPU: chạy cái này TRƯỚC, fail thì dừng
dvc repro                     # cả vòng: bakeoff → train → val → register → golden
cat reports/val_table.md reports/next_action.md

# Vòng sau (ví dụ): giảm epoch vì overfit
dvc exp run -S milestone.id=M2 -S 'milestone.note=giảm epoch' -S train.epochs=2
dvc exp show                  # so các vòng
```

**Xem MLflow** (tracking, registry, trace): `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
Trên máy thuê thì mở bằng `ssh -L 5000:localhost:5000 <máy thuê>`.
Có thể đặt `MLFLOW_TRACKING_URI` trỏ tới server MLflow của bạn để mọi máy log chung một chỗ.

**Dùng model** (mỗi lần gọi là một trace trong experiment `text2sql-inference`):
`python -m src.infer --db my.sqlite --question "Có bao nhiêu khách hàng ở Hà Nội?"`

**DVC remote** (để máy thuê và laptop dùng chung dữ liệu theo md5):
`dvc remote add -d storage s3://<bucket>/text2sql` (hoặc ssh/gdrive), sau đó `dvc push` / `dvc pull`.
Remote phải là **private** vì license ViText2SQL cấm phân phối lại.

## GPU
- sanity / train 0.8B: RTX 4090 24GB.
- bakeoff có ứng viên 2B full fine-tune: cần A100/H100 80GB, hoặc đặt `method: lora` cho ứng viên đó trong `params.yaml`.
- Ước tính cho grid mặc định (2 lr × 2 seed × 3 epoch) với 0.8B trên 4090: khoảng 6–10 giờ train + 1–2 giờ eval.
  Thu nhỏ grid nếu muốn rẻ hơn.

## License
ViText2SQL chỉ dùng cho nghiên cứu/giáo dục, **không phân phối lại**. Không public `data/`, model đã train hay DVC remote.
