# Model artifacts

The five files the application loads belong here:

| File | Size | Purpose |
| --- | --- | --- |
| `final_bin_best.pth` | 288 MiB | Binary clean/stego classifier |
| `final_multi_best.pth` | 288 MiB | Four-class payload classifier |
| `final_scaler.pkl` | 4.5 MiB | `StandardScaler` fitted on the feature vector |
| `final_bin_le.pkl` | <1 KiB | Binary label encoder (`clean`, `stego`) |
| `final_multi_le.pkl` | <1 KiB | Payload label encoder (`clean`, `html`, `js`, `ps`) |

They are not committed because they total 581 MiB. Download and verify them
with:

```sh
python scripts/fetch_models.py
```

The script checks each file against a pinned SHA-256 digest and is safe to
re-run.
