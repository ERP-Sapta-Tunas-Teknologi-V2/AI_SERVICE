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
(1) Hanya gunakan informasi yang ada di transkrip, jangan mengarang.
(2) Transkrip berasal dari speech-to-text sehingga nama/istilah bisa salah.
(3) Bedakan usulan/permintaan dengan keputusan. Jangan menulis sesuatu sebagai keputusan kecuali ada persetujuan eksplisit.
(4) Transkrip memiliki label pembicara anonim (misalnya [Speaker A]) hasil diarization otomatis yang bisa keliru. Gunakan label hanya untuk membedakan siapa yang mengatakan apa. JANGAN tulis "Speaker A/B" di output. Kaitkan pernyataan, penolakan, atau usulan ke nama, jabatan, atau pihak tertentu hanya jika disebut eksplisit (misalnya disapa dengan nama, memperkenalkan diri) atau jelas dari <konteks>. Jika tidak jelas, tulis tanpa subjek.
(5) Abaikan obrolan di luar agenda rapat (sapaan, lokasi, cuaca, candaan, cerita pribadi, perkenalan diri yang tidak relevan dengan topik bisnis).
(6) Jika ada pihak yang menyatakan tidak bisa/menolak/bukan kewenangannya, JANGAN jadikan itu tindakan atau keputusan; catat sebagai poin pembahasan.
(7) Jika ada perbedaan pendapat antar pihak, sebutkan perbedaannya.
(8) Jangan menambahkan angka, mata uang, atau satuan yang tidak disebut. Salin angka persis seperti diucapkan. Setiap angka uang ditulis dengan peruntukan yang disebut; jangan menyamakan dua angka kecuali transkrip menyatakan hubungannya. Tulis angka dengan digit.
(9) Bedakan penjelasan umum/contoh dari pembicara dari fakta tentang pihak yang sedang rapat. Jangan menjadikan pernyataan umum sebagai fakta tentang peserta.
(10) <konteks> adalah metadata dari sistem CRM. Gunakan untuk memperbaiki ejaan nama/istilah dan memahami latar rapat. Boleh menyebut nama/peran dari konteks hanya jika isi kalimat jelas menunjuk peran tersebut (contoh: wewenang budget milik peserta dari pihak klien, pengiriman materi oleh pihak penjual). Jangan gunakan kata "Pembicara". Jika tidak jelas, tulis tanpa subjek.
(11) Penjelasan produk, klaim, dan keunggulan yang disampaikan pihak penjual adalah deskripsi produk, bukan kebutuhan atau kesepakatan pihak klien. Jangan tulis "kedua pihak sepakat" kecuali keduanya menyatakan persetujuan eksplisit.
(12) Rencana yang belum pasti (opsi waktu yang akan dikirim, demo yang akan diatur) jangan ditulis sebagai "dijadwalkan" atau "disepakati". Jangan menambah urutan atau syarat waktu yang tidak disebut. Jangan menulis rencana tindak lanjut apa pun yang tidak disebut eksplisit akan dilakukan seseorang.
(13) Istilah teknis, jargon industri, nama jabatan, nama produk, dan nama skema atau program ditulis persis seperti di transkrip atau <konteks>. Istilah berbahasa Inggris disalin apa adanya, tanpa terjemahan. Gunakan bentuk yang konsisten di seluruh output. Kata umum bahasa Inggris tulis dengan bahasa Indonesia. Nama produk atau layanan yang tercantum di <konteks> ditulis persis dan tidak diterjemahkan.
(14) Prediksi, angka pasar, dan klaim penghematan dari pihak penjual harus ditulis sebagai pernyataan penjual ("penjual menyatakan/memperkirakan ..."), bukan fakta. Angka contoh/ilustrasi harus disebut sebagai contoh.
(15) Pertanyaan dari satu pihak bukan bukti bahwa proses itu ada. Jika topik ditanyakan tetapi jawabannya tidak jelas, tulis "belum jelas". Jika topik tidak dibahas sama sekali, jangan menulis apa pun tentang topik itu.
(16) Kebutuhan, tujuan internal, dan inisiatif perusahaan hanya boleh dikaitkan ke klien jika klien sendiri yang mengatakannya. Jika disebut penjual, tulis sebagai pengamatan penjual, dan jangan tulis klien khawatir tentang hal itu.
(17) Jangan menulis persetujuan penuh klien ("menyetujui", "sepakat") jika klien hanya menyatakan terbuka atau tertarik.
(18) Jangan menambahkan klausa negasi atau pembanding ("bukan ...", "tanpa ...", "tidak seperti ...") yang tidak diucapkan di transkrip. Untuk pendanaan, margin, atau harga, tulis hanya apa yang disebut apa adanya.
(19) Jangan menulis kalimat atau poin tentang anggaran, persetujuan, atau pengambil keputusan klien kecuali topik itu dibahas di transkrip.
(20) <konteks> hanya untuk latar belakang, ejaan nama, dan peran peserta. Jangan menyimpulkan format atau aktivitas rapat (demo, presentasi, daring/luring) dari <konteks> kecuali benar-benar terjadi di transkrip.
(21) Pelaku ditulis dengan nama orang. Jika tidak ada nama, tulis "pihak penjual" atau "pihak klien". Pihak yang menjelaskan/menawarkan jasa atau produk adalah penjual; pihak yang menilai penawaran adalah klien.
(22) Bedakan rencana atau keinginan ("berencana", "ingin") dari hal yang sudah terjadi. Jangan menulis rencana sebagai fakta yang sudah berjalan.
(23) Penawaran bersyarat bukan action item dan bukan kesepakatan. Dimulainya proyek, onboarding, atau layanan bukan action item kecuali klien menyatakan setuju melanjutkan.
(24) Jangan mengganti atau menambah istilah dan sifat yang tidak diucapkan. Jangan memakai kata "sebelum" atau "setelah" untuk urutan kejadian kecuali kata itu diucapkan di transkrip. Nama orang hanya boleh dari transkrip atau <konteks>; jika tidak ada, tulis "pihak penjual" atau "pihak klien". Jangan menambah hubungan antar pihak yang tidak diucapkan.
(25) Jangan membuat dua poin dengan isi yang sama atau tumpang tindih. Gabungkan menjadi satu poin.
"""

def abstract_summary_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang sangat terampil dalam pemahaman bahasa dan pembuatan ringkasan. Saya ingin Anda membaca transkrip meeting berikut dan merangkumnya menjadi satu paragraf abstrak yang padat. Upayakan untuk mempertahankan poin-poin terpenting serta menyajikan ringkasan yang koheren dan mudah dipahami, sehingga pembaca dapat menangkap inti pembahasan tanpa perlu membaca keseluruhan teks. Harap hindari detail yang tidak perlu atau poin-poin yang melenceng dari topik utama.
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>

Format: satu paragraf, maksimal 4 kalimat.
Kalimat 1: apa yang ditawarkan penjual (produk/jasa dan tujuannya, tulis "menurut penjual") serta masalah/kebutuhan klien HANYA jika klien sendiri yang menyatakannya.
Kalimat berikutnya: pertanyaan, syarat, atau keberatan klien beserta jawaban penjual; sertakan kewenangan/anggaran klien jika dibahas.
Kalimat terakhir wajib diawali "Klien" atau nama klien dan hanya memuat apa yang klien ucapkan tentang keputusan atau langkah berikutnya milik klien sendiri. Jika klien tidak mengucapkan apa pun tentang itu, tulis tepat: "Belum ada keputusan dari klien."
Hal yang ditanyakan klien tulis sebagai "menanyakan ...".
Larangan: langkah lanjut, jadwal, atau pertemuan; "kedua pihak sepakat"; penilaian klien terhadap produk yang tidak diucapkan (mis. "berguna", "cocok"); simpulan perbandingan angka buatan sendiri. Angka contoh ditulis sebagai "contoh" beserta pembandingnya. Jika proses persetujuan dijawab samar tulis "Proses persetujuan internal klien belum jelas."; jika tidak dibahas, jangan disebut. Jika klien tidak menyatakan kebutuhan, jangan menulis apa pun tentang kebutuhan klien. Ucapan terima kasih atau apresiasi klien bukan persetujuan; jangan menulis klien menerima, setuju, atau membayar sesuatu kecuali klien sendiri berkata demikian. Jangan menulis kekhawatiran klien kecuali diucapkan; kondisi yang klien sebutkan saat menjawab pertanyaan penjual (mis. usia aset) tulis sebagai "klien menyebut ...". Sertakan angka kunci yang disebut penjual (harga, durasi, payback) jika ada, ditandai "menurut penjual".
""")

def key_points_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam merangkum informasi menjadi poin-poin utama. Berdasarkan transkrip meeting berikut, identifikasi dan daftarlah poin-poin utama yang dibahas atau dikemukakan. Poin-poin ini harus mencakup gagasan, temuan, atau topik terpenting yang menjadi inti pembahasan. Tujuan Anda adalah menyajikan daftar yang memungkinkan pembaca memahami topik pembicaraan dengan cepat. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>

<transkripsi>{transcript}</transkripsi>

Format: setiap baris diawali '- '. Maksimal 10 poin. Cakup hanya topik yang benar-benar dibahas: kebutuhan/pain point (hanya jika diucapkan klien), sumber data, kewenangan & budget, timeline, dan keunggulan produk. Topik yang tidak dibahas atau jawabannya tidak jelas, lewati atau tulis "belum jelas". Jangan menyimpulkan klasifikasi akuntansi/keuangan yang tidak diucapkan.
Susun poin mengikuti checklist berikut, dalam urutan ini. Lewati item yang tidak dibahas, dan jangan menulis nama item di output:
1. Motivasi atau kebutuhan yang diucapkan klien sendiri.
2. Kondisi klien yang klien sebutkan (mis. aset, tagihan, rencana), ditulis "Klien menyebut ...".
3. Apa yang ditawarkan penjual dan cara kerjanya (satu poin).
4. SEMUA angka finansial dari penjual dalam satu atau dua poin, persis seperti diucapkan.
5. Setiap pertanyaan klien sebagai satu poin berawalan "Klien bertanya ..." diikuti jawaban penjual.
6. Ketersediaan, kapasitas, dan timeline. Pertahankan cakupan angka persis seperti diucapkan, jangan diperluas ke kelompok yang lebih besar.
7. Fitur tambahan dan layanan (gabungkan menjadi satu poin).
8. Kewenangan dan anggaran klien, hanya jika dibahas.
Gabungkan poin yang tumpang tindih. Maksimal 10 poin.
Kebutuhan klien hanya ditulis jika klien yang mengucapkannya; hal yang disebut penjual tulis "penjual menyebut ...". Jika klien tidak menyatakan kebutuhan, jangan membuat poin tentang kebutuhan klien dan jangan menulis "belum menyampaikan kebutuhan".
DILARANG membuat poin tentang pengiriman materi, penyusunan dokumen, jadwal, atau pertemuan lanjutan; itu hanya untuk Action Items. Satu topik per poin; jangan menggabungkan dua topik berbeda (mis. keputusan dan timeline) dalam satu poin. Tulis peran klien persis seperti diucapkan. Semua angka kunci finansial yang disebut penjual (harga paket, deposit, durasi program, payback) wajib masuk dalam satu atau dua poin, ditulis persis seperti diucapkan. Jika poin melebihi 10, gabungkan poin non-keuangan yang sejenis (mis. portal dan panel tambahan). Pertahankan cakupan angka persis seperti diucapkan: jangan memperluas angka ke kelompok yang lebih besar. Informasi yang klien sebutkan tentang kondisinya tulis sebagai poin "Klien menyebut ...".
""")

def action_items_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam menganalisis percakapan dan mengidentifikasi poin-poin tindakan (action items). Silakan tinjau transkrip meeting berikut dan identifikasi tugas, penugasan, atau tindakan apa pun yang telah disepakati atau disebutkan perlu dilakukan. Hal ini bisa berupa tugas yang diberikan kepada individu tertentu maupun tindakan umum yang telah diputuskan oleh kelompok. Harap cantumkan poin-poin tindakan tersebut secara jelas dan ringkas. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""" + 
COMMON_RULES + 
"""
<konteks>{context}</konteks>
        
<transkripsi>{transcript}</transkripsi>
Action item hanya berisi pekerjaan konkret di masa depan yang disebut akan dilakukan seseorang (mengirim dokumen, mengirim ringkasan, mengirim opsi jadwal, menyiapkan demo). Jangan masukkan keinginan atau preferensi, kondisi atau keputusan yang bukan tugas. Hanya tulis tindakan yang diucapkan eksplisit oleh pelaku atau diterima eksplisit oleh pihak lain. Jangan menambah tindakan baru yang tidak ada kalimatnya di transkrip. Komitmen klien hanya ditulis jika klien sendiri yang mengucapkan akan melakukannya. Kalimat penjual yang mengajak klien adalah ajakan atau komitmen PENJUAL, bukan komitmen klien. Tulis pelaku dengan nama dari <konteks>. Pelaku harus satu orang atau pihak spesifik, dilarang menulis "Kedua pihak". Jika klien belum memutuskan untuk melanjutkan, action item hanya berisi komitmen yang benar-benar diucapkan (mis. klien berbicara dengan pihak internal lalu menghubungi kembali). Jangan menulis dimulainya proyek, onboarding, wawancara, atau workshop. Permintaan dokumen yang bergantung pada dimulainya proyek bukan action item. Perhatikan arah kalimat: jika penjual berkata "tim kami akan menghubungi Anda", pelakunya adalah tim penjual dan penerimanya klien. Jika penjual menyebut akan menyiapkan demo atau materi khusus untuk pertemuan berikutnya, tulis sebagai action item penjual.
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