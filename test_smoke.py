"""รัน: python test_smoke.py : เช็กว่า UI สร้างได้ และ error ถูกต้องเมื่อไม่ใส่ไฟล์"""
import gradio as gr
import app

assert isinstance(app.demo, gr.Blocks)
for args in [(None, None, None, 0, "rmvpe", 0.5), ("a.wav", None, None, 0, "rmvpe", 0.5)]:
    try:
        app.convert(*args)
    except gr.Error:
        continue
    raise AssertionError("ต้อง raise gr.Error เมื่อขาดไฟล์")
print("ok")

# python test_smoke.py --full : ลองแยกเสียง+รวมกลับจริง (ต้องต่อเน็ตเพื่อโหลดโมเดลแยกเสียงครั้งแรก)
# ส่วนแปลงเสียง RVC ถูกแทนด้วยฟังก์ชันที่คืนไฟล์เดิม เพราะต้องใช้โมเดล .pth จริง
import sys

if "--full" in sys.argv:
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment

    sr = 44100
    t = np.arange(sr * 6) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write("test_song.wav", np.stack([x, x], 1), sr)
    app.convert = lambda vocals, *a: vocals
    mix, vocals, inst = app.cover("test_song.wav", "fake.pth", None, 0, "rmvpe", 0.5)
    assert abs(len(AudioSegment.from_file(mix)) - 6000) < 200, "ความยาวเพลงรวมผิด"
    print("full ok")
