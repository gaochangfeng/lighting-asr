# LASR: a lighting ASR model training platform
We believe the existing ASR training is too complex, so we want to design a ASR training platform as lighting as possible. 
With LASR, you can train your asr model by only providing the wav list and text list. 
All of the training deetails can be modified in the `config.yaml`.
You can also train your own torch model by warp it with the LASR interface. 

## The Recommended Configuration
LASR is based on python and pytorch, but we recommend the below configuration for use.
- Python3.7+  
- PyTorch 1.8.1+
- editdistance (eval the ASR results, install with pip)
- soundfile (if you want to read raw wav file or flac file, install with pip)
- librosa (if you want to read raw wav file or flac file, install with pip)
- jiwer (if you want to evaluate the asr results in python)
- [torchaudio](https://pytorch.org/) (to extract the speech feathers online)
- [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) (We recommand to use lighting to train the asr model, but you can also train the ASR model by yourself)

## Install Guideline

If all recommended configurations are satisfied, ETEH can be directly used by only adding it to the `PYTHONPATH` environment variable.
```
export PYTHONPATH=/path/to/lasrfolder/:$PYTHONPATH

```
## Use Guideline
### Prepare the training data
We use the Kaldi style scp file as input, `wav.scp` and `text` are needed. (the blank symbol is space)
#### Format of the wav.scp
```
wai_id_1 /path/to/auido1.wav
wai_id_2 /path/to/auido1.wav
```
#### Format of the text
```
wai_id_1 HELLO
wai_id_2 A NICE DAY
```
### Edit the config.yaml
In LASR, model, optimizer, criterion, tokenizer and even training data will be dymatic imported according to the `config.yaml`, in other word, you can import any python API by the format beyound:
```
model_config:
 name: torch.nn:Linear
 kwargs:
  in_features: 10
  in_features: 20

```
For the LASR training, you need to give the `model_config`, `opti_config`, `criterion_config`, `tokenizer_config`, `train_data_config` and `valid_data_config`. We also provide some API for using. If you want to use your own API, we suggest to use our API as the interface (especially for tokenizer and model). Details see the example.

### Training
You can use the `bin/train_lighting.py` for training.
### Decoding 
For decoding, you can define the some parameters in the `decode.yaml`.

You can use the `bin/decode_lighting.py` to evaluate your model for evaluation. But if you only want to recognize some audio, you can just use the `lasr.process.asrprocess.ASRProcess` as follow:
```
from lasr.process.asrprocess import ASRProcess

train_config="/path/to/train_config.yaml" 
decode_config="/path/to/decode_config.yaml"
model_path="/path/to/model.ckpt"
asrpipeline = ASRProcess(
    train_config=train_config, 
    decode_config=decode_config, 
    model_path=model_path
)
token, text = asrpipeline("test.wav")
print(token_1)
print(text_1)

```
