"""RVC Easy: อัปโหลดเสียง + โมเดล RVC (.pth) แล้วกดแปลง. ใช้ rvc-python เป็นเครื่องยนต์."""
import argparse
import os
import tempfile

import gradio as gr
import torch
from audio_separator.separator import Separator
from pydub import AudioSegment

from rvc_python.infer import RVCInference

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu:0"
SEP_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"  # แยกเสียงร้อง/ดนตรี ดาวน์โหลดอัตโนมัติครั้งแรก
_sep = None
_rvc = None  # สร้างครั้งเดียว: ตอนสร้างจะโหลด hubert/rmvpe (ดาวน์โหลดครั้งแรกครั้งเดียว)


def model_version(pth_path):
    # ไฟล์ .pth ของ RVC เก็บเวอร์ชันไว้ในตัว; ไม่มีค่า = v1
    return torch.load(pth_path, map_location="cpu").get("version", "v1")


def convert(audio_path, pth_path, index_path, pitch, method, index_rate):
    global _rvc
    if not audio_path:
        raise gr.Error("กรุณาอัปโหลดไฟล์เสียงก่อน")
    if not pth_path:
        raise gr.Error("กรุณาอัปโหลดไฟล์โมเดล .pth ก่อน")
    if _rvc is None:
        _rvc = RVCInference(device=DEVICE)
    _rvc.set_params(f0method=method, f0up_key=int(pitch), index_rate=float(index_rate))
    _rvc.load_model(pth_path, version=model_version(pth_path), index_path=index_path or "")
    out = os.path.join(tempfile.mkdtemp(), "converted.wav")
    _rvc.infer_file(audio_path, out)
    return out


def cover(song_path, pth_path, index_path, pitch, method, index_rate):
    """เพลงเต็ม → แยกเสียงร้อง/ดนตรี → แปลงเสียงร้อง → รวมกลับ."""
    global _sep
    if not song_path or not pth_path:
        raise gr.Error("กรุณาอัปโหลดเพลงและไฟล์โมเดล .pth ก่อน")
    work = tempfile.mkdtemp()
    if _sep is None:
        _sep = Separator(output_dir=work, model_file_dir="sep_models", output_format="WAV")
        _sep.load_model(SEP_MODEL)
    _sep.output_dir = work
    stems = {("Vocals" if "(Vocals)" in f else "Instrumental"): os.path.join(work, os.path.basename(f))
             for f in _sep.separate(song_path)}
    vocals = convert(stems["Vocals"], pth_path, index_path, pitch, method, index_rate)
    mix = AudioSegment.from_file(stems["Instrumental"]).overlay(AudioSegment.from_file(vocals))
    cover_path = os.path.join(work, "cover.wav")
    mix.export(cover_path, format="wav")
    return cover_path, vocals, stems["Instrumental"]


def settings():
    pitch = gr.Slider(-12, 12, value=0, step=1, label="ปรับระดับเสียง (semitone) ชาย→หญิง +12, หญิง→ชาย -12")
    method = gr.Radio(["rmvpe", "harvest", "crepe", "pm"], value="rmvpe", label="วิธีหา pitch (rmvpe แนะนำ)")
    index_rate = gr.Slider(0, 1, value=0.5, step=0.05, label="ความแรงของ index")
    return pitch, method, index_rate


def model_files():
    pth = gr.File(label="โมเดล (.pth)", file_types=[".pth"], type="filepath")
    index = gr.File(label="ไฟล์ .index (ไม่บังคับ แต่ทำให้เสียงเหมือนขึ้น)", file_types=[".index"], type="filepath")
    return pth, index


with gr.Blocks(title="RVC Easy") as demo:
    gr.Markdown(f"# RVC Easy: แปลงเสียงด้วยโมเดล RVC\nกำลังใช้: **{DEVICE}**")
    with gr.Tab("แปลงเสียง"):
        audio = gr.Audio(label="ไฟล์เสียงต้นฉบับ", type="filepath")
        pth, index = model_files()
        pitch, method, index_rate = settings()
        out = gr.Audio(label="ผลลัพธ์", type="filepath")
        gr.Button("แปลงเสียง", variant="primary").click(
            convert, [audio, pth, index, pitch, method, index_rate], out)
    with gr.Tab("ทำเพลงคัฟเวอร์"):
        gr.Markdown("อัปโหลด **เพลงเต็ม** ระบบจะแยกเสียงร้อง/ดนตรี → แปลงเสียงร้อง → รวมกลับให้เอง")
        song = gr.Audio(label="เพลงเต็ม", type="filepath")
        pth2, index2 = model_files()
        pitch2, method2, index_rate2 = settings()
        cover_out = gr.Audio(label="เพลงคัฟเวอร์ (รวมแล้ว)", type="filepath")
        vocals_out = gr.Audio(label="เฉพาะเสียงร้องที่แปลงแล้ว", type="filepath")
        inst_out = gr.Audio(label="เฉพาะดนตรี", type="filepath")
        gr.Button("ทำเพลงคัฟเวอร์", variant="primary").click(
            cover, [song, pth2, index2, pitch2, method2, index_rate2], [cover_out, vocals_out, inst_out])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="สร้างลิงก์สาธารณะ (ใช้บน Colab)")
    demo.queue().launch(share=p.parse_args().share)
