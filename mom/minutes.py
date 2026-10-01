import re
import time
import math
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from .logging import log_time, log_model

MAX_TOKEN = 5000
USE_FACTS = False

llm = ChatOllama(model="qwen3.5:9b", temperature=0, num_ctx=12288, reasoning=False)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")  # hanya tokenizer yang di-download, bukan model

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def check_ctx(prompt, transcript, context, label):
    n = count_tokens(prompt.format(transcript=transcript, context=context))
    if n > 10500:
        print(f"[WARN] {label}: prompt {n} token (num_ctx 12288)", flush=True)

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
    check_ctx(prompt, transcript, context, label)
    start = time.perf_counter()
    result = (prompt | llm).invoke({"transcript": transcript, "context": context}).content
    log_time(label, time.perf_counter() - start)
    return result

def chain_stream(prompt, transcript, label="chain_stream", context="-"):
    check_ctx(prompt, transcript, context, label)
    start = time.perf_counter()
    for chunk in (prompt | llm).stream({"transcript": transcript, "context": context}):
        if chunk.content:
            yield chunk.content
    log_time(label, time.perf_counter() - start)

COMMON_RULES = """
Aturan:
1. Hanya pakai informasi di transkrip, jangan mengarang. Transkrip hasil speech-to-text sehingga ejaan nama/istilah bisa salah; perbaiki dengan <konteks>.
2. Jangan tulis label "Speaker A/B". Pelaku ditulis dengan nama (dari transkrip atau <konteks>); jika tidak ada, tulis "pihak penjual" (yang menawarkan) atau "pihak klien" (yang menilai). Jangan menambah hubungan antar pihak yang tidak diucapkan.
3. Abaikan obrolan di luar agenda (sapaan, cuaca, candaan, cerita pribadi).
4. Usulan bukan keputusan. "Terbuka/tertarik" bukan setuju. Penolakan atau "bukan kewenangan" dicatat sebagai pembahasan, bukan tindakan. Perbedaan pendapat disebut.
5. Klaim, prediksi, dan angka dari penjual ditulis "penjual menyatakan ...". Kebutuhan/tujuan klien hanya jika klien sendiri yang mengatakannya. Penjelasan umum atau angka contoh bukan fakta tentang peserta; tandai "contoh".
6. Angka disalin persis dalam digit, sertakan peruntukannya. Jangan menyamakan dua angka, dan jangan menambah mata uang/satuan.
7. Istilah teknis, nama produk/program, dan istilah Inggris ditulis persis seperti di transkrip atau <konteks>, konsisten, tanpa terjemahan.
8. Rencana/keinginan bukan kejadian, jadwal, atau kesepakatan. Jangan menambah urutan waktu, kata "sebelum/setelah", atau klausa negasi/pembanding yang tidak diucapkan.
9. Topik (anggaran, pengambil keputusan, proses persetujuan) hanya ditulis jika dibahas. Jika ditanyakan tapi jawabannya tidak jelas, tulis "belum jelas". <konteks> bukan bukti format rapat (demo, daring/luring).
10. Jangan membuat poin ganda atau tumpang tindih.
"""

def abstract_summary_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang sangat terampil dalam pemahaman bahasa dan pembuatan ringkasan. Saya ingin Anda membaca transkrip meeting berikut dan merangkumnya menjadi satu paragraf abstrak yang padat. Upayakan untuk mempertahankan poin-poin terpenting serta menyajikan ringkasan yang koheren dan mudah dipahami, sehingga pembaca dapat menangkap inti pembahasan tanpa perlu membaca keseluruhan teks. Harap hindari detail yang tidak perlu atau poin-poin yang melenceng dari topik utama.
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>

Format:
- Kalimat 1: yang ditawarkan penjual ("menurut penjual"); kebutuhan klien hanya jika klien yang menyatakan.
- Berikutnya: pertanyaan/syarat/keberatan klien dan jawaban penjual; kewenangan/anggaran jika dibahas; angka kunci penjual (harga, durasi, payback) ditandai "menurut penjual".
- Kalimat terakhir diawali "Klien" dan hanya memuat keputusan/langkah berikutnya yang klien ucapkan. Jika tidak ada, tulis tepat: "Belum ada keputusan dari klien."
Larangan: langkah lanjut/jadwal, "kedua pihak sepakat", penilaian klien yang tidak diucapkan, ucapan terima kasih dianggap persetujuan.
""")

def key_points_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam merangkum informasi menjadi poin-poin utama. Berdasarkan transkrip meeting berikut, identifikasi dan daftarlah poin-poin utama yang dibahas atau dikemukakan. Poin-poin ini harus mencakup gagasan, temuan, atau topik terpenting yang menjadi inti pembahasan. Tujuan Anda adalah menyajikan daftar yang memungkinkan pembaca memahami topik pembicaraan dengan cepat. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>

Format: setiap baris diawali '- ', maksimal 10 poin, satu topik per poin. Urutan (lewati yang tidak dibahas, jangan tulis nama item):
1. Kebutuhan yang diucapkan klien.
2. Kondisi yang klien sebutkan ("Klien menyebut ...").
3. Yang ditawarkan penjual dan cara kerjanya (satu poin).
4. Semua angka finansial penjual (harga, deposit, durasi, payback) dalam 1-2 poin, persis seperti diucapkan.
5. Tiap pertanyaan klien: "Klien bertanya ..." + jawaban penjual.
6. Ketersediaan, kapasitas, timeline (cakupan angka tidak diperluas).
7. Fitur tambahan dan layanan (satu poin).
8. Kewenangan dan anggaran klien, jika dibahas.
Dilarang membuat poin tentang pengiriman materi, dokumen, jadwal, atau pertemuan lanjutan (itu untuk Action Items).
""")

def action_items_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam menganalisis percakapan dan mengidentifikasi poin-poin tindakan (action items). Silakan tinjau transkrip meeting berikut dan identifikasi tugas, penugasan, atau tindakan apa pun yang telah disepakati atau disebutkan perlu dilakukan. Hal ini bisa berupa tugas yang diberikan kepada individu tertentu maupun tindakan umum yang telah diputuskan oleh kelompok. Harap cantumkan poin-poin tindakan tersebut secara jelas dan ringkas. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>

Action item = pekerjaan konkret di masa depan yang pelakunya sendiri ucapkan akan dilakukan (mengirim dokumen/ringkasan/opsi jadwal, menyiapkan demo, menghubungi kembali) atau yang diterima eksplisit pihak lain.
- Bukan action item: keinginan, preferensi, kondisi, penawaran bersyarat, dimulainya proyek/onboarding/workshop.
- Komitmen klien hanya jika klien sendiri yang mengucapkan. Ajakan penjual kepada klien adalah komitmen penjual.
- Perhatikan arah: "tim kami akan menghubungi Anda" berarti pelakunya tim penjual.
- Pelaku satu pihak spesifik (nama dari <konteks>), jangan "Kedua pihak".
Format: tiap baris '- [Pelaku] akan [tindakan]'. Jika tidak ada, tulis "- Tidak ada".
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

FACTS_PROMPT = ChatPromptTemplate.from_template("""
Dari transkrip rapat berikut, tulis daftar singkat berlabel pihak. Gunakan <konteks> untuk menentukan siapa PENJUAL dan siapa KLIEN. Satu baris per butir, hanya dari ucapan di transkrip, jangan menambah atau menyimpulkan.
Label yang boleh dipakai:
[KLIEN BERTANYA] ... | [PENJUAL BERTANYA] ... | [PENJUAL MENJAWAB] ... | [KLIEN MENJAWAB] ... | [PENJUAL MENAWARKAN] ... | [PENJUAL BERKOMITMEN] ... | [KLIEN BERKOMITMEN] ... | [KLIEN MENYATAKAN] ...
Salin angka dan istilah persis seperti di transkrip. Tandai angka contoh dengan (contoh).

<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>
""")

def with_facts(chunk, context):
    facts = chain(FACTS_PROMPT, chunk, "facts", context)
    return f"<fakta_per_pihak>\n{facts}\n</fakta_per_pihak>\n\n{chunk}"

VERIFY_ACTIONS = ChatPromptTemplate.from_template("""
Periksa daftar action item di <hasil> terhadap <transkripsi>. Perbaiki daftar dengan aturan berikut:
- Hapus item yang tidak punya kalimat pendukung di transkrip.
- Hapus pernyataan keinginan/minat (mis. "menyatakan keinginan", "tertarik") karena bukan tindakan.
- Periksa arah: siapa yang berkata akan melakukan apa, kepada siapa. Jika suatu pihak berkata "tim kami akan menghubungi Anda", pelakunya tim pihak itu, bukan orangnya sendiri.
- Hapus item klien yang sebenarnya hanya ajakan atau komitmen penjual.
- Tulis pelaku dengan nama dari <konteks>, bukan "pihak penjual".
- Gabungkan duplikat. Jangan menambah item baru.
Keluarkan hanya daftar akhir, setiap baris diawali '- '. Jika kosong, tulis "- Tidak ada".

<konteks>{context}</konteks>
<transkripsi>{transcript}</transkripsi>
<hasil>{result}</hasil>
""")

def keep_action_lines(text):
    lines = [l for l in text.splitlines() if l.strip()]
    kept = [
        l for l in lines
        if re.search(r"\bakan\b|\bmenawarkan\b|Tidak ada", l)
        and not re.search(r"[\[\]]", l)
    ]
    return "\n".join(kept) or "- Tidak ada"

def verify_actions(draft, transcript, context):
    start = time.perf_counter()
    out = (VERIFY_ACTIONS | llm).invoke(
        {"transcript": transcript, "context": context, "result": draft}
    ).content
    log_time("action_items | verify", time.perf_counter() - start)
    return keep_action_lines(out)

FOLLOWUP_RE = re.compile(
    r"mengirim(kan)?\b|menjadwalkan|opsi waktu|sesi (demo )?lanjutan|pertemuan lanjutan",
    re.IGNORECASE,
)

def drop_followup_points(text):
    lines = [l for l in text.splitlines() if l.strip()]
    kept = []
    for l in lines:
        if FOLLOWUP_RE.search(l):
            print(f"[DROP key_point] {l}", flush=True)  # untuk review, hapus jika sudah yakin
        else:
            kept.append(l)
    return "\n".join(kept)

VERIFY_POINTS = ChatPromptTemplate.from_template("""
Periksa setiap poin di <hasil> terhadap <transkripsi>. Untuk tiap poin cek:
- Angka dan jumlah uang: apakah peruntukannya benar (harga paket, deposit, sisa pembayaran, tarif, durasi)? Jangan menyamakan atau menggabungkan dua angka kecuali transkrip menyatakan hubungannya.
- Siapa yang melakukan, memiliki, atau menyediakan sesuatu (penjual, klien, utilitas, pihak ketiga). Jangan menambah hubungan antar pihak (mitra, afiliasi) yang tidak diucapkan.
- Kata atau sifat yang tidak diucapkan (mis. "jangka panjang", "terbukti").
- Kalimat yang maknanya terbalik atau menggabungkan dua kejadian berbeda.
- Istilah produk atau skema yang diterjemahkan: kembalikan ke istilah asli di transkrip.
Perbaiki poin yang salah, hapus poin yang tidak didukung transkrip, jangan menambah poin baru, pertahankan format '- '. Keluarkan hanya daftar akhir.

<konteks>{context}</konteks>
<transkripsi>{transcript}</transkripsi>
<hasil>{result}</hasil>
""")

def verify_points(draft, transcript, context):
    start = time.perf_counter()
    out = (VERIFY_POINTS | llm).invoke(
        {"transcript": transcript, "context": context, "result": draft}
    ).content
    log_time("key_points | verify", time.perf_counter() - start)
    new = [l for l in out.splitlines() if l.strip().startswith("- ")]
    old = [l for l in draft.splitlines() if l.strip().startswith("- ")]
    if len(new) < max(1, len(old) // 2):   # jaga-jaga jika verifier merusak daftar
        return draft
    return "\n".join(new)

def generate_minutes(transcript, context="-"):
    log_model(MAX_TOKEN, llm.model)

    chunks = add_overlap(split(transcript))

    raw_chunks = chunks
    if USE_FACTS:
        chunks = [with_facts(c, context) for c in chunks]

    prompts = [
        ("abstract_summary", abstract_summary_prompt()),
        ("key_points", key_points_prompt()),
        ("action_items", action_items_prompt()),
    ]

    for key, prompt in prompts:
        if len(chunks) == 1:
            if key == "action_items":
                draft = chain(prompt, chunks[0], "action_items | draft", context)
                yield {"type": "mom", "key": key, "content": verify_actions(draft, chunks[0], context)}
            elif key == "key_points":
                text = chain(prompt, chunks[0], "key_points | chunk 1/1", context)
                text = verify_points(drop_followup_points(text), raw_chunks[0], context)
                yield {"type": "mom", "key": key, "content": text}
            else:
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