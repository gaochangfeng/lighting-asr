import librosa
import soundfile
import numpy
import os
def try_read_kaldi(file_io):
    if file_io is not None:
        line = file_io.readline().strip()
        if line == "":
            file_io.close()
            return "None", "None"
        return line.split(' ')[0], " ".join(line.split(' ')[1:])
    else:
        return "None", "None"
    
def read_audio(wav_path):
    if os.path.splitext(wav_path)[1] in ['.wav', '.flac']:
        return read_wav_soundfile(wav_path)
    elif os.path.splitext(wav_path)[1] in ['.mp3']:
        return read_mp3_librosa(wav_path)
    else:
        raise ValueError("Unknown wav type for " + wav_path)
    
def read_wav_soundfile(wav_path):
    waveform, sample_rate = soundfile.read(wav_path)
    return waveform, sample_rate

def read_mp3_librosa(mp3_path):
    waveform, sample_rate = librosa.load(mp3_path)
    return waveform, sample_rate

# def ffmpeg_read(bpayload: bytes, sampling_rate=16000):
#     """
#     Helper function to read an audio file through ffmpeg.
#     """
#     import subprocess
#     ar = f"{sampling_rate}"
#     ac = "1"
#     format_for_conversion = "f32le"
#     ffmpeg_command = [
#         "ffmpeg",
#         "-i",
#         "pipe:0",
#         "-ac",
#         ac,
#         "-ar",
#         ar,
#         "-f",
#         format_for_conversion,
#         "-hide_banner",
#         "-loglevel",
#         "quiet",
#         "pipe:1",
#     ]
#     print(bpayload)
#     try:
#         with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
#             output_stream = ffmpeg_process.communicate(bpayload)
#     except FileNotFoundError as error:
#         raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
#     out_bytes = output_stream[0]
#     print(output_stream)
#     audio = numpy.frombuffer(out_bytes, numpy.float32)
#     if audio.shape[0] == 0:
#         raise ValueError("Malformed soundfile")
#     return audio, sampling_rate

def read_kaldi_feats1(file_path):
    from kaldi_io import read_mat
    return read_mat(file_path)


def get_audio_duration(file_path):
    return librosa.get_duration(filename = file_path)

def get_audio_samplerate(file_path):
    return librosa.get_samplerate(file_path)

def read_list(path):
    with open(path,'r',encoding='utf-8') as f:
        char_list = f.read().splitlines()
    return char_list

def dict_reader(path,sc=' ',append=True, eos='<eos>'):
    world_dict = {}
    last = 0
    with open(path,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()
    for line in lines:
        key, value = line.split(sc)[0], int(line.split(sc)[1])
        world_dict[key] = value
        last = value + 1
    if append:
        world_dict[eos] = last
    return world_dict
