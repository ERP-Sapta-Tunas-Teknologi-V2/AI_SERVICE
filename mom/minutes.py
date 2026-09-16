from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen2.5:14b", temperature=0, think=False)

def chain(prompt, transcript):
    chain = prompt | llm
    for chunk in chain.stream({"transcript": transcript}):
        if chunk.content:
            yield chunk.content

def abstract_summary_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang sangat terampil dalam pemahaman bahasa dan pembuatan ringkasan. Saya ingin Anda membaca transkrip meeting berikut dan merangkumnya menjadi satu paragraf abstrak yang padat. Upayakan untuk mempertahankan poin-poin terpenting serta menyajikan ringkasan yang koheren dan mudah dipahami, sehingga pembaca dapat menangkap inti pembahasan tanpa perlu membaca keseluruhan teks. Harap hindari detail yang tidak perlu atau poin-poin yang melenceng dari topik utama.

<transkripsi>{transcript}</transkripsi>

Anda adalah AI yang sangat terampil dalam pemahaman bahasa dan pembuatan ringkasan. Saya ingin Anda membaca transkrip meeting berikut dan merangkumnya menjadi satu paragraf abstrak yang padat. Upayakan untuk mempertahankan poin-poin terpenting serta menyajikan ringkasan yang koheren dan mudah dipahami, sehingga pembaca dapat menangkap inti pembahasan tanpa perlu membaca keseluruhan teks. Harap hindari detail yang tidak perlu atau poin-poin yang melenceng dari topik utama.
""")

def key_points_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam merangkum informasi menjadi poin-poin utama. Berdasarkan transkrip meeting berikut, identifikasi dan daftarlah poin-poin utama yang dibahas atau dikemukakan. Poin-poin ini harus mencakup gagasan, temuan, atau topik terpenting yang menjadi inti pembahasan. Tujuan Anda adalah menyajikan daftar yang memungkinkan pembaca memahami topik pembicaraan dengan cepat. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
        
<transkripsi>{transcript}</transkripsi>

Anda adalah AI yang ahli dalam merangkum informasi menjadi poin-poin utama. Berdasarkan transkrip meeting berikut, identifikasi dan daftarlah poin-poin utama yang dibahas atau dikemukakan. Poin-poin ini harus mencakup gagasan, temuan, atau topik terpenting yang menjadi inti pembahasan. Tujuan Anda adalah menyajikan daftar yang memungkinkan pembaca memahami topik pembicaraan dengan cepat. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""")

def action_items_prompt():
    return ChatPromptTemplate.from_template("""
Anda adalah AI yang ahli dalam menganalisis percakapan dan mengidentifikasi poin-poin tindakan (action items). Silakan tinjau transkrip meeting berikut dan identifikasi tugas, penugasan, atau tindakan apa pun yang telah disepakati atau disebutkan perlu dilakukan. Hal ini bisa berupa tugas yang diberikan kepada individu tertentu maupun tindakan umum yang telah diputuskan oleh kelompok. Harap cantumkan poin-poin tindakan tersebut secara jelas dan ringkas. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
        
<transkripsi>{transcript}</transkripsi>

Anda adalah AI yang ahli dalam menganalisis percakapan dan mengidentifikasi poin-poin tindakan (action items). Silakan tinjau transkrip meeting berikut dan identifikasi tugas, penugasan, atau tindakan apa pun yang telah disepakati atau disebutkan perlu dilakukan. Hal ini bisa berupa tugas yang diberikan kepada individu tertentu maupun tindakan umum yang telah diputuskan oleh kelompok. Harap cantumkan poin-poin tindakan tersebut secara jelas dan ringkas. Hanya berikan poin-poinnya, jangan berikan kalimat seperti "berikut".
""")

def generate_minutes(transcript):
    prompts = [
        ("abstract_summary", abstract_summary_prompt()),
        ("key_points", key_points_prompt()),
        ("action_items", action_items_prompt()),
    ]

    for key, prompt in prompts:
        for chunk in chain(prompt, transcript):
            yield {"type": "mom", "key": key, "content": chunk}
        yield {"type": "mom_done", "key": key}
    yield {"type": "done"}