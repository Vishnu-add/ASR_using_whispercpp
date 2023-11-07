import gradio as gr
import soundfile as sf
import tempfile
import shutil
import os
import librosa
import time
import numpy as np
import subprocess 

# command = r"""wine './whisper_blas_bin_v1_3_0/main.exe' -h"""
# wine_command = """sudo apt-get install wine"""
command2 = """chmod +777 ./whisper_blas_bin_v1_3_0/main.exe"""
# wine_c = subprocess.run(wine_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
perm = subprocess.run(command2, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
# result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# print("Wine Installation: ",wine_c)
print("Access Installation: ",perm)
# Fpr win32 instalattion while using medium model
# command3 = "apt install sudo"
# command4 = "dpkg --add-architecture i386"
# command5 = "apt-get update"
# command6 = "apt-get install wine32:i386"
# t1= subprocess.run(command3, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
# t2= subprocess.run(command4, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
# t3= subprocess.run(command5, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
# t4= subprocess.run(command6, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
# print("T1: ",t1)
# print("T2: ",t2)
# print("T3: ",t3)
# print("T4: ",t4)


def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr = 16000)
    return y_resampled

def transcribe(audio):
    sr,y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y_resampled = resample_to_16k(y, sr)
    
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        sf.write(temp_audio_path, y_resampled, 16000)


    
    # command = rf"""wine './whisper_blas_bin_v1_3_0/main.exe' -m './whisper_blas_bin_v1_3_0/models/ggml-model-whisper-small.en.bin' -osrt -f '{temp_audio_path}' -nt"""  # English only
    command = rf"""wine './whisper_blas_bin_v1_3_0/main.exe' -m './whisper_blas_bin_v1_3_0/models/ggml-model-whisper-base.bin' -osrt -f '{temp_audio_path}' -nt"""    # Multilingual
    # win32 error while using medium model
    # command = rf"""wine './whisper_blas_bin_v1_3_0/main.exe' -m './whisper_blas_bin_v1_3_0/models/ggml-model-whisper-medium-q5_0.bin' -osrt -f '{temp_audio_path}' -nt"""    # Multilingual
    
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
    inputs = "microphone",
    # gr.Audio(sources=["microphone"]),
    outputs = [gr.Textbox(label="CLI_Transcription"),gr.Textbox(label="Time taken for Transcription")],
    examples=["./Samples/Hindi_1.mp3","./Samples/Hindi_2.mp3","./Samples/Tamil_1.mp3","./Samples/Tamil_2.mp3","./Samples/Marathi_1.mp3","./Samples/Marathi_2.mp3","./Samples/Nepal_1.mp3","./Samples/Nepal_2.mp3","./Samples/Telugu_1.wav","./Samples/Telugu_2.wav","./Samples/Malayalam_1.wav","./Samples/Malayalam_2.wav","./Samples/Gujarati_1.wav","./Samples/Gujarati_2.wav","./Samples/Bengali_1.wav","./Samples/Bengali_2.wav"]

)

demo.launch()