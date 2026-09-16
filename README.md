Modul AI yang menyediakan layanan **Minutes of Meeting (MoM)** untuk CRM dan **Chatbot** untuk clocking.

Saat ini, modul yang tersedia adalah **MoM**. Modul Chatbot akan dikembangkan kemudian.

# Table of Contents

* [Project Structure](#project-structure)
* [Minutes of Meeting (MoM)](#minutes-of-meeting-mom)
  * [Tech Stack](#tech-stack)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [How to Run](#how-to-run)
  * [Testing](#testing)
    * [Test via index.html](#test-via-indexhtml)
    * [Test API CRM](#test-api-crm)

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
* **OpenAI Whisper** — Transkripsi audio (speech-to-text)
* **Whisper Turbo** — Model untuk transkripsi
* **Ollama** — Local LLM runer
* **Qwen 2.5 14B** — LLM untuk generate Minutes of Meeting

## Prerequisites

Pastikan aplikasi berikut sudah terinstall:
* Python
* FFmpeg
* Ollama
* Git

### FFmpeg

Digunakan oleh Whisper untuk memproses file audio.

Jika menggunakan Windows dan Chocolatey (https://chocolatey.org/), instal dengan:

```powershell
choco install ffmpeg
```

### Ollama

Instal Ollama mengikuti dokumentasi resmi [Ollama Quickstart](https://docs.ollama.com/quickstart).

Setelah Ollama terinstal, download model:

```powershell
ollama pull qwen2.5:14b
```

Untuk melihat model yang tersedia:

```powershell
ollama list
```

## Installation

Clone repository:

```powershell
git clone <repo-url>
cd AI_SERVICE
```

Semua command berikut dijalankan dari folder `AI_SERVICE/`.

### 1. Membuat virtual environment

```powershell
python -m venv .venv
```

Aktifkan virtual environment. Jika di PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Jika berhasil, terminal akan menunjukkan `(.venv)`.

### 2. Instal PyTorch

Untuk GPU NVIDIA dengan CUDA 12.6:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Periksa instalasi PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Jika menggunakan GPU dan instalasi berhasil, `torch.cuda.is_available()` seharusnya menghasilkan `True`.

### 3. Instal OpenAI Whisper

```powershell
pip install -U openai-whisper
```

### 4. Instal dependency project lainnya

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
