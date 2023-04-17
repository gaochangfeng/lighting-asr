import yaml
import torch
from lasr.data.tokenizer import CharTokenizer
from lasr.data.reader import read_audio
from lasr.data.datatrans import register_trans
from lasr.utils.generater import BaseConfig
from lasr.decode.ctc_att_decoder import CTC_ATT_Decoder

class ASRProcess(object):
    def __init__(self, train_config, decode_config, model_path, device="cpu") -> None:
        self.device = device
        with open(train_config) as f:
            self.train_config = yaml.safe_load(f) 
        with open(decode_config) as f:
            self.decode_config = yaml.safe_load(f) 

        # token_type = self.decode_config["test_data_config"]["kwargs"]["tokenizer"]
        # token_dict = self.decode_config["test_data_config"]["kwargs"]["dict_path"]
        # if token_type == "char":
        #     self.tokenizer = CharTokenizer(token_dict, sc='')
        # else:
        #     self.tokenizer = CharTokenizer(token_dict, sc=' ')

        model_config = self.train_config['model_config'] 
        tokenizer_config = self.train_config["tokenizer_config"]
        # model_config["kwargs"]["odim"] = self.tokenizer.dict_size()
        # print(model_config)
        # exit()

        model = BaseConfig(**model_config).generateExample()
        self.tokenizer = BaseConfig(**tokenizer_config).generateExample()
        checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']        
        state_dict = {k.split('.', maxsplit=1)[1]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)

        asr_decode_config = self.decode_config["decode_config"]
        self.decoder_ctc_att = CTC_ATT_Decoder(
                model, self.tokenizer.ID_VALUE_SOS, self.tokenizer.ID_VALUE_EOS,
                beam=asr_decode_config["beam"], ctc_beam=asr_decode_config["ctc_beam"], 
                ctc_weight=asr_decode_config["ctc_weight"],
                rnnlm=None, lm_weight=asr_decode_config["lm_rate"]
        )

        self.audio_trans = self.decode_config["test_data_config"]["kwargs"]["audio_trans"]

    def frontend(self, wav_path):
        wav_array, sample_rate = read_audio(wav_path)
        wav_array = register_trans["avgchannel"](wav_array)
        if sample_rate != 16000:
            wav_array = register_trans["resample:16k"](wav_array, sample_rate)
        for trans in self.audio_trans:
            wav_array = register_trans[trans](wav_array)
        return wav_array

    def backend(self, ans):
        hypo = self.tokenizer.decode(ans)
        return hypo

    def model_forward(self, feats):
        x = torch.as_tensor(feats).to(self.device)
        xlen = torch.LongTensor([len(x)]).to(x.device)
        with torch.no_grad():
            ans = self.decoder_ctc_att.decode_feat(x, xlen)
        ans = ans[0]['yseq'][1:-1]
        return ans

    def __call__(self, wav, decode_type = "ctc_att"):
        wav_array = self.frontend(wav)
        ans = self.model_forward(wav_array)
        hypo = self.backend(ans)
        return hypo
    
if __name__ == '__main__':
    train_config="/data/gaochangfeng/docker/project_final/env/eteh_light/example/lasr/exp/train_aishell12_hkust_cmv_conformer_spec_len1000/lightning_logs/version_0/hparams.yaml"
    decode_config="/data/gaochangfeng/docker/project_final/env/eteh_light/example/lasr/conf/decode_aishell.yaml"
    model_path="/data/gaochangfeng/docker/project_final/env/eteh_light/example/lasr/exp/train_aishell12_hkust_cmv_conformer_spec_len1000/lightning_logs/version_0/checkpoints/average_best_10.ckpt"
    asrpipeline = ASRProcess(
        train_config=train_config, 
        decode_config=decode_config, 
        model_path=model_path
    )
    text_1 = asrpipeline("/mnt/ssd/gaochangfeng/wav/AIShell1/data_aishell/wav/train/S0002/BAC009S0002W0122.wav")
    print(text_1)
