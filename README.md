Modul AI yang menyediakan layanan **Minutes of Meeting (MoM)** untuk CRM dan **Chatbot** untuk clocking.

Saat ini, modul yang tersedia adalah **MoM**. Modul Chatbot akan dikembangkan kemudian.

# Table of Contents

* [Project Structure](#project-structure)
* [Minutes of Meeting (MoM)](#minutes-of-meeting-mom)
  * [Tech Stack](#tech-stack)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [How to Run](#how-to-run)
  * [Deployment](#deployment)
  * [Testing](#testing)

# Project Structure

```text
AI_SERVICE/
│
├── app.py
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── mom/
│   ├── __init__.py
│   ├── routes.py
│   ├── service.py
│   ├── transcribe.py
│   ├── minutes.py
│   └── requirements.txt
│
├── chatbot/
│   └── __init__.py
│
├── templates/
│   └── index.html
│
└── temp_audio/
```

# Minutes of Meeting (MoM)

Modul MoM menangani:
1. Upload file audio (tipe file `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, atau `webm`)
2. Transkripsi audio menggunakan Whisper
3. Analisis transkrip dan generate Minutes of Meeting menggunakan Qwen
4. Menampilkan hasil melalui `index.html`
5. Menyediakan API untuk CRM

## Tech Stack

* **Flask** — Web framework dan REST API
* **Faster Whisper (Turbo)** — Transkripsi audio (speech-to-text)
* **Ollama** — Local LLM runer
* **Qwen 2.5 14B** — LLM untuk generate Minutes of Meeting

## Prerequisites

Pastikan aplikasi berikut sudah terinstall:
* Python
* PyTorch
* cuBLAS & cuDNN
* Ollama

### PyTorch

Instal mengikuti dokumentasi resmi [PyTorch](https://pytorch.org/get-started/locally/).

Untuk GPU NVIDIA dengan CUDA 12.6:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Periksa instalasi PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Jika menggunakan GPU dan instalasi berhasil, `torch.cuda.is_available()` seharusnya menghasilkan `True`.

### cuBLAS & cuDNN

Install library NVIDIA melalui pip:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Cari file DLL yang terinstall:

```powershell
Get-ChildItem -Path .\.venv\Lib\site-packages\nvidia -Recurse -Filter "cublas64_12.dll"                           
Get-ChildItem -Path .\.venv\Lib\site-packages\nvidia -Recurse -Filter "cudnn64_9.dll"
```

Catat lokasi folder `bin` yang berisi masing-masing DLL.

Kemudian tambahkan folder tersebut ke `PATH` Windows menggunakan PowerShell **Administrator**:

```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "Machine") + ";<cublas_bin_path>;<cudnn_bin_path>",
    "Machine"
)
```

Tutup dan buka kembali PowerShell setelah mengubah `PATH`.

### Ollama

Instal mengikuti dokumentasi resmi [Ollama](https://docs.ollama.com/quickstart).

Setelah Ollama terinstal, download model:

```powershell
ollama pull qwen2.5:14b
```

Untuk melihat model yang tersedia:

```powershell
ollama list
```

## Installation

### 1. Clone repository

```powershell
git clone <repo-url>
cd AI_SERVICE
```

Semua command berikut dijalankan dari folder `AI_SERVICE/`.

### 2. Buat virtual environment

```powershell
python -m venv .venv
```

Aktifkan virtual environment. Jika di PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Jika berhasil, terminal akan menunjukkan `(.venv)`.

### 3. Instal dependency project lainnya

```powershell
pip install -r mom/requirements.txt
```

## How to Run

### 1. Jalankan Ollama

Pastikan Ollama sedang berjalan.

Periksa:

```text
http://localhost:11434
```

Pastikan model `qwen2.5:14b` tersedia di:

```powershell
ollama list
```

### 2. Jalankan Flask

Dari root `AI_SERVICE/` dan virtual environment yang sudah aktif:

```powershell
flask run --debug
```

Jika berhasil, Flask akan berjalan di:

```text
http://127.0.0.1:5000
```

## Deployment

Panduan berikut untuk deploy AI Service ke server **Ubuntu** menggunakan Gunicorn + Nginx + systemd.

### 1. Update sistem & instal dependency dasar

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git nginx
```

### 2. Instal NVIDIA Driver & CUDA (jika menggunakan GPU)

Pastikan driver NVIDIA dan CUDA toolkit sudah terinstall sesuai kebutuhan PyTorch (CUDA 12.6).

```bash
nvidia-smi
```

Pastikan perintah di atas menampilkan info GPU. Jika belum, instal driver NVIDIA terlebih dahulu mengikuti dokumentasi resmi NVIDIA untuk Ubuntu.

### 3. Instal Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b
```

Jalankan Ollama sebagai service (biasanya otomatis terinstall sebagai systemd service):

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

### 4. Clone repository & setup virtual environment

```bash
cd /opt
sudo git clone <repo-url> ai_service
cd ai_service
sudo python3 -m venv .venv
source .venv/bin/activate
```

### 5. Instal PyTorch & dependency

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
pip install -r mom/requirements.txt
pip install gunicorn
```

Verifikasi PyTorch mendeteksi GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 6. Jalankan dengan Gunicorn (systemd service)

Buat file service:

```bash
sudo nano /etc/systemd/system/ai_service.service
```

Isi dengan:

```ini
[Unit]
Description=AI Service (MoM) - Gunicorn
After=network.target ollama.service

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/ai_service
Environment="PATH=/opt/ai_service/.venv/bin"
ExecStart=/opt/ai_service/.venv/bin/gunicorn --workers 2 --bind 127.0.0.1:8000 --timeout 300 app:app

Restart=always

[Install]
WantedBy=multi-user.target
```

> `--timeout 300` diperlukan karena proses transkripsi & generate MoM bisa memakan waktu lama.
> Jumlah `--workers` disesuaikan dengan kapasitas GPU/VRAM server (biasanya 1-2 worker untuk beban model AI).

Aktifkan dan jalankan service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai_service
sudo systemctl start ai_service
sudo systemctl status ai_service
```

### 7. Setup Nginx sebagai reverse proxy

```bash
sudo nano /etc/nginx/sites-available/ai_service
```

Isi dengan:

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    client_max_body_size 200M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

> `client_max_body_size` perlu diperbesar agar upload file audio meeting tidak ditolak Nginx.

Aktifkan konfigurasi:

```bash
sudo ln -s /etc/nginx/sites-available/ai_service /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 8. (Opsional) Setup HTTPS dengan Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain
```

### 9. Verifikasi deployment

```bash
curl -X POST -F "audio=@/path/to/meeting.mp3" http://your_domain_or_ip/api/mom
```

Jika berhasil, service dapat diakses melalui domain/IP server tanpa perlu menjalankan `flask run` secara manual.

### Update aplikasi (redeploy)

```bash
cd /opt/ai_service
sudo git pull
source .venv/bin/activate
pip install -r mom/requirements.txt
sudo systemctl restart ai_service
```

## Testing

### A. Test via index.html

#### 1. Buka:

```text
http://127.0.0.1:5000
```

#### 2. Upload file audio meeting.

#### 3. Aplikasi akan menjalankan:

```text
Audio
  ↓
Whisper
  ↓
Transcript
  ↓
Qwen 2.5 14B
  ↓
Minutes of Meeting
```

#### 4. Pada `index.html`, hasil yang ditampilkan adalah:

* Transcript
* MoM
  * Abstract Summary
  * Key Points
  * Action Items

### B. Test API CRM

API yang digunakan oleh CRM Django:

```http
POST /api/mom
```

Full endpoint:

```text
http://127.0.0.1:5000/api/mom
```

API menerima file audio menggunakan `multipart/form-data` dengan field `audio`.

#### Menggunakan cURL

Contoh di PowerShell:

```powershell
curl.exe -X POST `
  -F "audio=@D:\path\to\meeting.mp3" `
  http://127.0.0.1:5000/api/mom
```

#### Response

Untuk CRM, API hanya mengirimkan MoM.

Transkrip hanya digunakan secara internal oleh AI Service dan ditampilkan pada `index.html`.

### Alur API ke CRM

```text
CRM Django
    │
    │ POST /api/mom
    │ audio = meeting.mp3
    ▼
AI_SERVICE
    │
    ▼
Whisper
    │
    ▼
Transcript
    │
    ▼
Qwen 2.5 14B
    │
    ▼
MoM
    │
    ▼
CRM Django
```

File audio yang di-upload disimpan sementara di:

```text
temp_audio/
```

dan akan dihapus setelah proses selesai.