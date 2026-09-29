import re
import time
import math
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from .timing_log import log_time, log_model

MAX_TOKEN = 5000

llm = ChatOllama(model="qwen3.5:9b", temperature=0, num_ctx=12288, reasoning=False)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")  # hanya tokenizer yang di-download, bukan model

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def split(transcript, max_token=MAX_TOKEN):
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    counts = [count_tokens(s) for s in sentences]
    total = sum(counts)

    n_chunks = max(1, math.ceil(total / max_token))
    target = total / n_chunks

    chunks, current, current_tokens = [], [], 0
    for sentence, n in zip(sentences, counts):
        if current and len(chunks) < n_chunks - 1 and current_tokens + n / 2 > target:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n
    if current:
        chunks.append(" ".join(current))
    return chunks

def add_overlap(chunks, n_sentences=3):
    result = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = " ".join(re.split(r'(?<=[.!?])\s+', prev)[-n_sentences:])
        result.append(tail + " " + cur)
    return result

def chain(prompt, transcript, label="chain", context="-"):
    start = time.perf_counter()
    result = (prompt | llm).invoke({"transcript": transcript, "context": context}).content
    log_time(label, time.perf_counter() - start)
    return result

def chain_stream(prompt, transcript, label="chain_stream", context="-"):
    start = time.perf_counter()
    for chunk in (prompt | llm).stream({"transcript": transcript, "context": context}):
        if chunk.content:
            yield chunk.content
    log_time(label, time.perf_counter() - start)

COMMON_RULES = """
Aturan:
(1) Hanya gunakan informasi yang ada di transkrip, jangan mengarang.
(2) Transkrip berasal dari speech-to-text sehingga nama/istilah bisa salah.
(3) Bedakan usulan/permintaan dengan keputusan. Jangan menulis sesuatu sebagai keputusan kecuali ada persetujuan eksplisit.
(4) Transkrip tidak memiliki label pembicara. Jangan mengaitkan pernyataan, penolakan, atau usulan ke orang, jabatan, atau instansi tertentu kecuali disebut eksplisit di kalimat yang sama. Jika tidak jelas, tulis tanpa subjek.
(5) Abaikan obrolan di luar agenda rapat (sapaan, lokasi, cuaca, candaan, cerita pribadi, perkenalan diri yang tidak relevan dengan topik bisnis).
(6) Jika ada pihak yang menyatakan tidak bisa/menolak/bukan kewenangannya, JANGAN jadikan itu tindakan atau keputusan; catat sebagai poin pembahasan.
(7) Jika ada perbedaan pendapat antar pihak, sebutkan perbedaannya.
(8) Jangan menambahkan angka, mata uang, atau satuan yang tidak disebut eksplisit. Salin angka persis seperti di transkrip (tanpa simbol mata uang).
(9) Bedakan penjelasan umum/contoh dari pembicara dari fakta tentang pihak yang sedang rapat. Jangan menjadikan pernyataan umum sebagai fakta tentang peserta.
(10) <konteks> adalah metadata dari sistem CRM. Gunakan untuk memperbaiki ejaan nama/istilah dan memahami latar rapat. Boleh menyebut nama/peran dari konteks hanya jika isi kalimat jelas menunjuk peran tersebut (contoh: wewenang budget milik peserta dari pihak klien, pengiriman materi oleh pihak penjual). Jangan gunakan kata "Pembicara". Jika tidak jelas, tulis tanpa subjek.
(11) Penjelasan produk, klaim, dan keunggulan yang disampaikan pihak penjual adalah deskripsi produk, bukan kebutuhan atau kesepakatan pihak klien. Jangan tulis "kedua pihak sepakat" kecuali keduanya menyatakan persetujuan eksplisit.
(12) Rencana yang belum pasti (opsi waktu yang akan dikirim, demo yang akan diatur) jangan ditulis sebagai "dijadwalkan" atau "disepakati". Jangan menambah urutan atau syarat waktu yang tidak disebut.
"""

def abstract_summary_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang sangat terampil dalam pemahaman bahasa dan pembuatan ringkasan. Saya ingin Anda membaca transkrip meeting berikut dan merangkumnya menjadi satu paragraf abstrak yang padat. Upayakan untuk mempertahankan poin-poin terpenting serta menyajikan ringkasan yang koheren dan mudah dipahami, sehingga pembaca dapat menangkap inti pembahasan tanpa perlu membaca keseluruhan teks. Harap hindari detail yang tidak perlu atau poin-poin yang melenceng dari topik utama.

<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>
""" + COMMON_RULES + """
Format: satu paragraf, maksimal 4 kalimat. Sebutkan kebutuhan klien, pendekatan pihak penjual (dipisahkan dari kebutuhan klien), kewenangan budget, dan langkah lanjut yang belum pasti.
""")

def key_points_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam merangkum informasi menjadi poin-poin utama. Berdasarkan transkrip meeting berikut, identifikasi dan daftarlah poin-poin utama yang dibahas atau dikemukakan. Poin-poin ini harus mencakup gagasan, temuan, atau topik terpenting yang menjadi inti pembahasan. Tujuan Anda adalah menyajikan daftar yang memungkinkan pembaca memahami topik pembicaraan dengan cepat. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".

<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>
""" + COMMON_RULES + """
Format: setiap baris diawali '- '. Maksimal 8 poin. Pastikan mencakup: kebutuhan/pain point, sumber data yang dibahas, kewenangan & budget, timeline (termasuk jika tidak ada timeline ketat), dan keunggulan utama produk yang dijelaskan.
Jangan masukkan tugas tindak lanjut (pengiriman dokumen, penjadwalan, isi demo); itu sudah ada di bagian Action Items.
""")

def action_items_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam menganalisis percakapan dan mengidentifikasi poin-poin tindakan (action items). Silakan tinjau transkrip meeting berikut dan identifikasi tugas, penugasan, atau tindakan apa pun yang telah disepakati atau disebutkan perlu dilakukan. Hal ini bisa berupa tugas yang diberikan kepada individu tertentu maupun tindakan umum yang telah diputuskan oleh kelompok. Harap cantumkan poin-poin tindakan tersebut secara jelas dan ringkas. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".

<konteks>{context}</konteks>
        
<transkripsi>{transcript}</transkripsi>
""" + COMMON_RULES + """
Action item hanya berisi pekerjaan konkret di masa depan yang disebut akan dilakukan seseorang (mengirim dokumen, mengirim ringkasan, mengirim opsi jadwal, menyiapkan demo). Jangan masukkan keinginan atau preferensi (contoh: "ingin melihat produk dulu sebelum melibatkan CIO"), kondisi ("jika biaya melebihi..."), atau keputusan yang bukan tugas.
Format: setiap baris diawali '- ' dengan pola "[Pelaku] akan [tindakan]". Jika tidak ada tindakan, tulis "- Tidak ada".
""")

REDUCE_INSTRUCTIONS = {
    "abstract_summary": "Gabungkan ringkasan parsial berikut menjadi SATU paragraf abstrak maksimal 5 kalimat. Jangan menambah informasi, jangan mengulang, dan jangan menyebut pembicara jika tidak eksplisit. Jangan menulis ada kesepakatan atau keputusan kecuali tertulis eksplisit di data. Jika belum ada keputusan final, tulis bahwa belum ada keputusan final, lalu sebutkan opsi yang diperdebatkan dan pihak yang keberatan.",
    "key_points": "Gabungkan daftar poin berikut, hapus duplikasi, dan pastikan poin dari SEMUA bagian tercakup. Maksimal 12 poin, format '- '. Langsung tulis poin tanpa kalimat pembuka atau penutup.",
    "action_items": "Gabungkan daftar berikut, hapus duplikasi dan item yang bukan tindakan (pernyataan aturan, penjelasan, atau kata 'boleh/diperbolehkan'). Jangan ubah usulan menjadi keputusan. Jangan menambah item yang tidak ada di data. Setiap baris diawali '- ' lalu kalimat tindakan yang spesifik, tanpa label 'tindakan'. Langsung tulis poin tanpa kalimat pembuka atau penutup.",
}

def reduce_prompt(key):
    return ChatPromptTemplate.from_template(
        REDUCE_INSTRUCTIONS[key] + "\n\n<konteks>{context}</konteks>\n\n<data>{transcript}</data>"
    )

def generate_minutes(transcript, context="-"):
    log_model(MAX_TOKEN, llm.model)

    chunks = add_overlap(split(transcript))
    prompts = [
        ("abstract_summary", abstract_summary_prompt()),
        ("key_points", key_points_prompt()),
        ("action_items", action_items_prompt()),
    ]

    for key, prompt in prompts:
        if len(chunks) == 1:
            for piece in chain_stream(prompt, chunks[0], f"{key} | chunk 1/1", context):
                yield {"type": "mom", "key": key, "content": piece}
        else:
            # MAP
            partials = [
                chain(prompt, c, f"{key} | map chunk {i}/{len(chunks)}", context)
                for i, c in enumerate(chunks, 1)
            ]
            combined = "\n\n".join(partials)

            # REDUCE
            for piece in chain_stream(reduce_prompt(key), combined, f"{key} | reduce", context):
                yield {"type": "mom", "key": key, "content": piece}

        yield {"type": "mom_done", "key": key}
    yield {"type": "done"}