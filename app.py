import gradio as gr
import soundfile as sf
import tempfile
import shutil
import os
import librosa
import time
import numpy as np
import subprocess 


def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr = 16000)
    return y_resampled

def transcribe(audio,):
    sr,y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y_resampled = resample_to_16k(y, sr)
    
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        sf.write(temp_audio_path, y_resampled, 16000)

    command = rf"""'./whisper_blas_bin_v1_3_0/main.exe' -m './whisper_blas_bin_v1_3_0/models/ggml-model-whisper-small.en.bin' -osrt -f '{temp_audio_path}' -nt"""
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()
    print("Output",result.stdout)
    print("Error",result.stderr)
    transcription = result.stdout
    print(transcription)
    
    print("--------------------------")
    print(f"Execution time: {end_time - start_time} seconds")
    return transcription, (end_time - start_time)



demo = gr.Interface(
    transcribe,
    gr.Audio(source="microphone"),
    # gr.Audio(sources=["microphone"]),
    outputs=[gr.Textbox(label="CLI_Transcription"),gr.Textbox(label="Time taken for Transcription")]
)

demo.launch()
