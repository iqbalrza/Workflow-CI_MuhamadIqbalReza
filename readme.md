# Workflow CI - Muhamad Iqbal Reza

Submission **Membangun Sistem Machine Learning (MSML)** - Dicoding x Microsoft Elevate.

## Kriteria 3: Membuat Workflow CI

Repository ini berisi MLflow Project dan GitHub Actions CI/CD pipeline untuk re-training otomatis model klasifikasi **Telco Customer Churn**.

---

## Struktur Folder

```
Workflow-CI_MuhamadIqbalReza/
├── .github/
│   └── workflows/
│       └── ci-train.yml              # GitHub Actions CI pipeline
├── MLProject/
│   ├── MLProject                     # MLflow Project definition
│   ├── conda.yaml                    # Environment dependencies
│   ├── modelling.py                  # Training script
│   └── telco_churn_preprocessing/    # Dataset siap latih
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── DockerHub.txt                     # URL Docker Hub image
└── README.md
```

---

## CI/CD Pipeline

Workflow otomatis trigger saat ada push ke branch `main` yang mengubah file di folder `MLProject/`.

### Steps Pipeline (Advanced)

| Step | Deskripsi |
|---|---|
| Checkout | Clone repository ke runner |
| Set up Python 3.12.7 | Setup environment Python |
| Check Env | Verifikasi Python, pip, Docker tersedia |
| Install dependencies | Install MLflow, scikit-learn, dagshub, dll |
| Set MLflow Tracking URI | Konfigurasi DagsHub sebagai tracking server |
| Run mlflow project | Jalankan training via `mlflow run` |
| Get latest MLflow run_id | Ambil run_id terbaru untuk build Docker |
| Install Python dependencies | Install boto3 untuk upload |
| Upload to GitHub | Simpan artefak MLflow ke GitHub Actions |
| Build Docker Model | Build Docker image via `mlflow models build-docker` |
| Log in to Docker Hub | Autentikasi ke Docker Hub menggunakan secrets |
| Tag Docker Image | Tag image dengan `latest` dan commit SHA |
| Push Docker Image | Push ke Docker Hub registry |

---

## MLflow Project

File `MLProject` mendefinisikan entry point training:

```yaml
name: telco_churn_msml
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 200}
      max_depth: {type: str, default: "10"}
    command: python modelling.py
```

Cara menjalankan lokal:
```bash
mlflow run ./MLProject --env-manager=local -P n_estimators=200 -P max_depth=10
```

---

## Docker Image

Model di-containerize menggunakan `mlflow models build-docker` dan di-push ke Docker Hub.

Docker Hub: lihat `DockerHub.txt`

Cara pull dan serve image:
```bash
docker pull USERNAME/telco-churn:latest
docker run -p 5001:8080 USERNAME/telco-churn:latest
```

Setelah container berjalan, model bisa diakses via REST API:
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

---

## GitHub Secrets

Workflow menggunakan 4 secrets (tidak ada credentials yang hardcode):

| Secret | Kegunaan |
|---|---|
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_TOKEN` | Access token Docker Hub |
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_TOKEN` | Access token DagsHub |

---

## Environment

- Python 3.12.7
- mlflow==2.19.0
- scikit-learn==1.8.0
- pandas==2.3.3
- numpy==2.4.4
- dagshub, matplotlib, seaborn, joblib