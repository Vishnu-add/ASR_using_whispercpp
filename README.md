# ASR_using_whispercpp

Run
```
pip install -r requirements.txt
```

## Download the whispercpp files
- Download_link : https://github.com/ggerganov/whisper.cpp/releases/tag/v1.3.0
- Navigate to Assests and download the desired files for your system
  - whisper-bin-Win32.zip
  - whisper-bin-x64.zip
  - whisper-blas-bin-Win32.zip
  - whisper-blas-bin-x64.zip

## Download the models
- Download_link : https://ggml.ggerganov.com/

![image](https://github.com/Add-Vishnu/ASR_using_whispercpp/assets/139844342/38b2f2d1-31b4-41d1-a024-fc7d7f860335)


## Run app.py

## Errors
- If you encounter any issue with gr.Audio uncomment the commented line and comment the existing one.
- The existing works with gradio


## Functions
- transcribe: This function takes the audio input resample it using resamplr_to_16k function and saves it in  a temporary .wav file which will be deted later. The command is stored in the command variable with the main.exe path and whisper's model path.
  ```
  command rf"""'./whisper_blas_bin_v1_3_0/main.exe' -m './whisper_blas_bin_v1_3_0/models/ggml-model-whisper-small.en.bin' -osrt -f '{temp_audio_path}' -nt"""
  ```
  - -m : Used to specify the models path
  - -osrt : Used to save the subtitles in a file
  - -f : Used to specify the audio path to be transcribed
  - -nt : Used to specify - no-timestamps
- resample_to_16k: This function resamples the speech rate of audio to 16k

## References:
- Main Github : https://github.com/ggerganov/whisper.cpp
- Files of WhisperCPP : https://github.com/ggerganov/whisper.cpp/releases/tag/v1.3.0
- Download Models from : https://ggml.ggerganov.com/
- Downlaod Models from Hugging face : https://huggingface.co/ggerganov/whisper.cpp
- https://github.com/ggerganov/whisper.cpp/tree/master/models
