import torch
import numpy
import logging
import lasr.data.reader as reader
from lasr.data.tokenizer import CharTokenizer
from lasr.data.datatrans import register_trans

def batch_list(array_list, pad_dim=0, pad_value=-1, dtype=numpy.float32):
    B = len(array_list)
    T = max([x.shape[pad_dim] for x in array_list])
    if len(array_list[0].shape) == 1:
        batch_shape = (B, T)
    elif len(array_list[0].shape) == 2:
        batch_shape = (B, T, *array_list[0].shape[:pad_dim], *array_list[0].shape[pad_dim+1:])
    else:
        batch_shape = (B, T, *array_list[0].shape[:pad_dim], *array_list[0].shape[pad_dim+1:])

    batch_array = pad_value * numpy.ones(batch_shape, dtype=dtype)
    for e, inp in enumerate(array_list):
        batch_array[e, :inp.shape[0]] = inp

    return batch_array


class AudioDataSet(torch.utils.data.Dataset):
    '''Basic DataSet inherit from torch.data.Dataset, self.train_set is a list to store Data, List[List[Tuple(Data,Label)]]
        length of the DataSet is the length of self.train_set. When use DataLoader to get Data, return a List of Tuple(Data,Lable)
    :param str name: the name the Dataset.
    '''

    def __init__(self,
            wav_list=None, 
            text_list=None, 
            feats_list=None, # 还没有实现这个功能
            tokenizer=None, #分为char, word, toknizer_path三种？
            audio_trans=("fbank:80"),
            feats_trans=None,
            pad_audio = 0,
            pad_feats = 0
        ):
        self.wav_list = wav_list
        self.text_list = text_list
        self.feats_list = feats_list
        self.pad_audio = pad_audio
        self.pad_feats = pad_feats
        self.audio_trans = audio_trans
        self.train_set = [
            
        ]
        self.tokenizer = tokenizer
        # if tokenizer == "char":
        #     self.tokenizer = CharTokenizer(dict_path, sc='')
        # else:
        #     self.tokenizer = CharTokenizer(dict_path, sc=' ')

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        item = self.train_set[index]
        return item

    def load_check_data(self):
        # print("I am checking data")
        self.load_dataset()
        self.check_dataset()

    # def empty_func(self):
    #     print("This is a trap")

    def load_dataset(self):
        if isinstance(self.wav_list, str) and isinstance(self.text_list, str):
            self.wav_list = [self.wav_list]
            self.text_list = [self.text_list]
            # self.feats_list = [self.feats_list]
        for i in range(len(self.wav_list)):
            wav_file = open(self.wav_list[i], 'r') if self.wav_list[i] else None
            text_file = open(self.text_list[i], 'r') if self.text_list[i] else None
            while True:
                id_set = set()
                wav_id, wav_path = reader.try_read_kaldi(wav_file)
                text_id, text_line = reader.try_read_kaldi(text_file)
                # feats_id, feats_path = reader.try_read_kaldi(feat_file)

                id_set.add("None")
                id_set.add(wav_id)
                id_set.add(text_id)
                # id_set.add(feats_id)
                if(len(id_set)>2):
                    raise RuntimeError("input data id doesn't match {},{}".format(wav_id, text_id))
                
                id_set.remove("None")
                if len(id_set)==0:
                    break
                id = list(id_set)[0]

                self.train_set.append(
                    {
                        "id": id,
                        "wav": wav_path,
                        "text": text_line,
                        "feats": "None"                    
                    }
                )

    def check_dataset(self):
        # from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        # pool = ProcessPoolExecutor()
        # print("check the audio duration")
        # futures1 = [pool.submit(reader.get_audio_duration, item["wav"]) for item in self.train_set]
        # results1 = [item.result() for item in tqdm(futures1)]
        # print("check the audio samplerate")
        # futures2 = [pool.submit(reader.get_audio_samplerate, item["wav"]) for item in self.train_set]
        # results2 = [item.result() for item in tqdm(futures2)]

        # for i in range(len(self.train_set)):
        logging.info("Checking data")
        for item in tqdm(self.train_set):
            if "wav" in item and item["wav"] != "None":
                wav_path = item["wav"]
                item["wav_len"] = reader.get_audio_duration(wav_path)
            else:
                item["wav_len"] = 0

            if "text" in item and item["text"] != "None":
                text = item["text"]
                item["token"], item["token_id"] = self.tokenizer.encode(text, add_sos_eos=False)
                item["token_id"] = numpy.array(item["token_id"])
                item["token_len"] = len(item["token_id"] )
            else:
                item["token_id"], item["token_len"] = numpy.array([0]), 0

    def example_data(self):
        return {
                # 在构造数据集时读取
                "id": "wav_id",
                "wav": "path/to/wav",       
                "text": "text",
                "feats": "path/to/feats",         
                # 在构造完数据集后读取
                "wav_len": 3,
                # "wav_sr": 16000,
                "token": ['t','e','x','t'],
                "token_len": 4,
                "token_id": [1,2,3,4],
                "feats_shape": [2,3],
                # 在生成batch时读取， 不会出现在dataset里
                # "wav_array": numpy(3),
                # "feats_array": numpy.zeros((2,3)),
                # 在生成batch时计算，不会出现在dataset里
                # "trans_wav_array": numpy(3),
                # "trans_feats_array": numpy.zeros((2,3)),                
            }

    def example_batch(self):
        return {
                "id": ["wav_id"],
                "wav": ["path/to/wav"],
                "wav_len": [3],
                "wav_array": [[0,0,0]],
                # "wav_sr": [16000],

                "text": ["text"],
                "token": [['t','e','x','t']],
                "token_len": [4],
                "token_id": [[1,2,3,4]],

                "feats": ["path/to/feats"],
                "feats_shape": [[2,3]],
                "feats_array": [numpy.zeros((2,3))],
            }

    # def load_dataarray(self):
    #     for i in range(len(self.train_set)):
    #         if "wav" in self.train_set[i] and self.train_set[i]["wav"] != "None":
    #             wav_path = self.train_set[i]["wav"]
    #             self.train_set[i]["wav_array"], _ = reader.read_wav_soundfile(wav_path)

    def MergeBatch(self, batch):
        new_batch = self.example_batch()
        for k in new_batch:
            new_batch[k].clear()
        for b in batch:
            for k in b:
                new_batch[k].append(b[k])

        new_batch["wav_len"].clear()
        for wav_path in new_batch["wav"]:
            if wav_path != "None":
                wav_array, sample_rate = reader.read_audio(wav_path)
                wav_array = register_trans["avgchannel"](wav_array)
                if sample_rate != 16000:
                    wav_array = register_trans["resample:16k"](wav_array, sample_rate)
                for trans in self.audio_trans:
                    wav_array = register_trans[trans](wav_array)
                    wav_len = wav_array.shape[0]
            else:
                wav_array = [0]
                wav_len = 0
            new_batch["wav_array"].append(wav_array)
            new_batch["wav_len"].append(wav_len)

        new_batch["wav_array"] = batch_list(new_batch["wav_array"], pad_value=self.pad_audio)
        new_batch["wav_len"] = numpy.array(new_batch["wav_len"], dtype=numpy.int64)

        for feats_path in new_batch["feats"]:
            if feats_path != "None":
                feats_array, _ = reader.read_kaldi_feats1(wav_path)
            else:
                feats_array = numpy.zeros((1,1))
            new_batch["feats_array"].append(feats_array)

        new_batch["feats_array"] = batch_list(new_batch["feats_array"], pad_value=self.pad_feats)

        new_batch["token_id"] =  batch_list(new_batch["token_id"], pad_value=self.tokenizer.ID_VALUE_PAD, dtype=numpy.int64)
        new_batch["token_len"] =  numpy.array(new_batch["token_len"], dtype=numpy.int64)

        return new_batch
        
    def collate_fn(self, batch):
        #batch是一个list，大小为batchsize，每一项都是getitem的结果，getitem得到的也是list，大小为json里的batchsize
        # m_bancth = []
        # for b in batch:
        #     m_bancth += b
        batch = self.MergeBatch(batch)
        for k in batch:
            if isinstance(batch[k], numpy.ndarray):
                batch[k] = torch.from_numpy(batch[k])

        return batch
    
class BatchAudioDataSet(AudioDataSet):
    def __init__(
            self, wav_list=None, text_list=None, feats_list=None, tokenizer="char", audio_trans=["fbank80"], feats_trans=None, pad_audio=0, pad_feats=0,
            batch_size = 32,
            batch_duration = 320,
            batch_bin = 32 * 500 * 80,
            batch_type = "size",
            max_duration = 30,
            min_duration = 0.3,
            text_freq = 0.08
        ):
        super().__init__(wav_list, text_list, feats_list, tokenizer, audio_trans, feats_trans, pad_audio, pad_feats)
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.batch_bin = batch_bin
        self.batch_duration = batch_duration
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.text_freq = text_freq

    def check_dataset(self):
        super().check_dataset()
        self.train_set.sort(key=lambda x: x["wav_len"] * 16000 + x["token_len"])
        before_filter = len(self.train_set)
        self.train_set = list(
            filter(
                # lambda x: x["wav_len"]<self.min_duration or x["wav_len"]>self.max_duration or x["wav_len"]/x["token_len"] < self.text_freq,
                lambda x: x["wav_len"]<self.max_duration and x["wav_len"]>self.min_duration and x["wav_len"]/x["token_len"] > self.text_freq,
                self.train_set        
            )
        )
        after_filter = len(self.train_set)
        # if before_filter != after_filter:
        # print("filer {} to {} utterances", before_filter, after_filter)
        # exit()
        if self.batch_type == "size":
            self.make_batch_size(self.batch_size)
        elif self.batch_type == "duration":
            self.make_batch_duration(self.batch_duration)

    def make_batch_size(self, size):
        self.train_set = [self.train_set[i:i+size] for i in range(0,len(self.train_set),size)]

    def make_batch_duration(self, duration):
        new_train_set =[]
        bg, ed = 0, 0
        batch_len = 0
        while ed < len(self.train_set):
            batch_len += self.train_set[ed]["wav_len"]
            ed += 1
            if batch_len >= duration:
                new_train_set.append(self.train_set[bg:ed])
                bg, ed = ed, ed
                batch_len = 0

        if bg != len(self.train_set):
            new_train_set.append(self.train_set[bg:]) 
        self.train_set = new_train_set

    def collate_fn(self, batch):
        #batch是一个list，大小为batchsize，每一项都是getitem的结果，getitem得到的也是list，大小为json里的batchsize
        m_batch = []
        for b in batch:
            m_batch += b
        return super().collate_fn(m_batch)
