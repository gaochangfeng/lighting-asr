# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
# Lingxuan Ye       yelingxuan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)
# export OMP_NUM_THREADS=1
import torch
import yaml
import argparse
import os
from lasr.utils.generater import BaseConfig
from lasr.utils.average_checkpoints import model_average
import editdistance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path")
    parser.add_argument("-train_config")
    parser.add_argument("-decode_config")
    parser.add_argument("-output_file")
    parser.add_argument("-dict", type=str, help="the number of the character")
    parser.add_argument("-device", type=str, default="cpu", help="the decode device")
    parser.add_argument("-avg", type=int, default=10, help="the index of the gpu")
    parser.add_argument("-choose", type=str, default="best", help="the index of the gpu")
    args = parser.parse_args()
   
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    with open(args.decode_config) as f:
        decode_config = yaml.safe_load(f)

    model_config = train_config['model_config'] 
    tokenizer_config = train_config["tokenizer_config"]
    test_data_config = decode_config['test_data_config']
    asr_decode_config = decode_config['decode_config'] 

    tokenizer = BaseConfig(**tokenizer_config).generateExample()
    test_dataset = BaseConfig(**test_data_config).generateExample(tokenizer=tokenizer)
    test_dataset.load_check_data()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_dataset.collate_fn
    )

    # output_dim = tokenizer.dict_size()
    # model_config["kwargs"]["odim"] = output_dim

    model = BaseConfig(**model_config).generateExample()

    if os.path.isfile(args.model_path):
        checkpoint = torch.load(
            args.model_path, map_location=lambda storage, loc: storage)    
        state_dict = checkpoint['state_dict']
        print("Decoding with the averged model:" + args.model_path)
    else:
        state_dict, choose = model_average(args.model_path, ids = args.choose, num = args.avg)
        torch.save({'state_dict':state_dict}, args.model_path + '/average_{}_{}.ckpt'.format(args.choose, args.avg))
        print("Decoding with the averged model from:")
        print(choose)
    state_dict = {k.split('.', maxsplit=1)[1]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # if args.gpu >= 0:
    #     if torch.cuda.device_count() <= args.gpu:
    #         print('Error!gpu is not enough')
    #         exit()
    #     else:
    #         device = args.gpu
            
    # else:
    #     device = "cpu" 
    model = model.to(args.device)
    print(model)

    decode_method = asr_decode_config["decode_method"] if "decode_method" in decode_config else "ctc_att"

    if decode_method.startswith("ctc_att"):
        from lasr.decode.ctc_att_decoder import CTC_ATT_Decoder
        decoder = CTC_ATT_Decoder(
                model, tokenizer.ID_VALUE_SOS, tokenizer.ID_VALUE_EOS,
                beam=asr_decode_config["beam"], ctc_beam=asr_decode_config["ctc_beam"], 
                ctc_weight=asr_decode_config["ctc_weight"],
                rnnlm=None, lm_weight=asr_decode_config["lm_rate"]
        )
    
    # elif decode_method == "ctc_bs":
    #     from eteh.tools.decode.ctc_bs_decoder import CTC_Decoder
    #     decoder = CTC_Decoder(
    #             beam_size=decode_config["beam"], ctc_beam=decode_config["ctc_beam"], 
    #             sos=model_config["odim"]-1,
    #             rnn_lm=lm_model, lm_rate=decode_config["lm_rate"]
    #     )
    # elif decode_method == "ctc_kenlm_lexcoin":
    #     from eteh.tools.decode.ctc_w2l_decoder import CTC_KenLM_Decoder
    #     import math
    #     decoder = CTC_KenLM_Decoder(
    #             beam_size=decode_config["beam"], beam_threshold=decode_config["beam_threshold"], 
    #             lexicon=decode_config["lexicon"], tokens_dict=decode_config["tokens_dict"], 
    #             kenlm_model=decode_config["kenlm_model"],
    #             sos='<eos>', blk='<blank>', unk='<unk>', sil=None,
    #             lm_weight=decode_config["lm_weight"], word_score=decode_config["word_score"], 
    #             unk_score=-math.inf, sil_score=decode_config["sil_score"],
    #             log_add=False,
    #     )
    # else:
    #     pass

    # decode_file = open(args.data_list, 'r')
    
    # if args.ref_list is not None:
    #     decode_ref = open(args.ref_list, 'r', encoding='utf-8')
    # else:
    #     decode_ref = None

    error = 0
    token_num = 0
    output_file = open(args.output_file, 'w', encoding='utf-8') 
    ref_list = []
    hyp_list = []

    for batch in test_dataloader:
        uid = batch["id"][0]
        feats = batch["wav_array"][0]
        print("id", uid) # , path, feats.shape)

        if decode_method == "ctc_att":
            hypo = ctc_att_decode(model, decoder, tokenizer.char_list, args.device, feats)
        elif decode_method == "ctc_att_online":
            hypo = ctc_att_decode_online(model, decoder, tokenizer.char_list, args.device, feats)
        else:
            hypo = ctc_bs_decode(model, decoder, tokenizer.char_list, args.device, feats)
        ref = batch["text"][0]
        _, hypo = tokenizer.decode(hypo, no_special=True)
        if ref != 'None':
            dist = editdistance.eval(hypo, ref)
        else:
            dist = 0
        print('ref:', ref)
        print('hyp:', hypo)
        print('dis:', dist)
        ref_list.append(ref)
        hyp_list.append(hypo)

        error += dist
        token_num += len(ref)
        output_file.write(hypo+" ({})\n".format(uid))

    print("Totol WER is {}".format(error/token_num))
    try:
        import jiwer

        out = jiwer.process_characters(
            ref_list,
            hyp_list,
        )
        print(jiwer.visualize_alignment(out))

        out = jiwer.process_words(
            ref_list,
            hyp_list,
        )
        print(jiwer.visualize_alignment(out))

    except ImportError:
        print("jiwer is not installed")

    # decode_file.close()
    # output_file.close()
    # if decode_ref: decode_ref.close()


def ctc_att_decode(model, decoder, char_list, device, feats):
    x = feats
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([len(x)]).to(x.device)
    with torch.no_grad():
        ans = decoder.decode_feat(x, xlen)
    ans = ans[0]['yseq'][1:-1]
    # ans = [char_list[uid] for uid in ans][1:-1]
    return ans


def ctc_att_decode_online(model, decoder, char_list, device, feats):
    x = feats
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([len(x)]).to(x.device)
    with torch.no_grad():
        ans = decoder.decode_feat_online(x, xlen)
    ans = ans[0]['yseq']
    ans = [char_list[uid] for uid in ans][1:-1]
    return str.join(' ', ans)


def ctc_bs_decode(model, decoder, char_list, device, feats):
    x = feats.unsqueeze(0)
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([x.shape[1]]).to(x.device)
    y = torch.LongTensor([len(char_list)-1]).cuda(device)
    ylen = [1]
    with torch.no_grad():
        model.eval()
        prob = model.get_ctc_prob(x, xlen)

    prob = torch.softmax(prob, -1)[0].detach().cpu().numpy()
    
    with torch.no_grad():
        ans_list = decoder.decode_problike(prob, True)
    ans = ans_list[0][0]
    ans = [char_list[uid] for uid in ans]
    return str.join(' ', ans).replace("<eos>", "")

        
if __name__ == '__main__':
    import sys
    print(' '.join(sys.argv))
    main()
