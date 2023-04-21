
import lasr.data.reader as reader
import logging
try:
    import tokenizers
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
except ImportError:
        logging.warning("tokenizers is not installed, some function could be banned")

try:
    import sentencepiece as spm
except ImportError:
        logging.warning("sentencepiece is not installed, some function could be banned")

class BaseTokenizer(object):
    ID_VALUE_BLACK = 0
    ID_VALUE_SOS = 1
    ID_VALUE_EOS = 2
    ID_VALUE_MASK = 3
    ID_VALUE_PAD = 4
    ID_VALUE_UNK = 5
    ID_VALUE_IGNORE = -1
    ID_KEY_BLACK = "<BLANK>"
    ID_KEY_SOS = "<SOS>"
    ID_KEY_EOS = "<EOS>"
    ID_KEY_MASK = "[MASK]"
    ID_KEY_PAD = "[PAD]"
    ID_KEY_UNK = "[UNK]"

    SPECIAL_VALUE = [
        ID_VALUE_BLACK,
        ID_VALUE_SOS,
        ID_VALUE_EOS,
        ID_VALUE_MASK,
        ID_VALUE_PAD,
        ID_VALUE_UNK,            
    ]

    SPECIAL_KEY = [
        ID_KEY_BLACK,
        ID_KEY_SOS,
        ID_KEY_EOS,
        ID_KEY_MASK,
        ID_KEY_PAD,
        ID_KEY_UNK,            
    ]
    def __init__(self):
        pass

    def get_token_id(self, token):
        raise NotImplementedError()

    def get_id_token(self, id):
        raise NotImplementedError()


    def encode(self, text, add_sos_eos=True):
        raise NotImplementedError()
    
    def decode(self, token_id, no_special=False):
        raise NotImplementedError()

    
    def dict_size(self):
        raise NotImplementedError()

class CharTokenizer(BaseTokenizer):
    def __init__(self, dict_path, sc='') -> None:
        super().__init__()
        self.char_list = [
            self.ID_KEY_BLACK,
            self.ID_KEY_SOS,
            self.ID_KEY_EOS,
            self.ID_KEY_MASK,
            self.ID_KEY_PAD,
            self.ID_KEY_UNK,
        ]
        self.sc = sc
        self.char_list += reader.read_list(dict_path)

        self.char_dict = {}
        for i,c in enumerate(self.char_list):
            self.char_dict[c] = i


    def get_token_id(self, token):
        token = token.upper()
        if token in self.char_dict:
            return self.char_dict[token]
        else:
            return self.char_dict[self.ID_KEY_UNK]

    def get_id_token(self, id):
        if id > len(self.char_list):
            return self.ID_KEY_UNK
        else:
            return self.char_list[id]


    def encode(self, text, add_sos_eos=True):
        sc = self.sc
        if len(sc) == 0:
            token = [c for c in text]
        else:
            token = text.split(sc)
        if add_sos_eos:
            token = [self.ID_KEY_SOS] + token + [self.ID_KEY_EOS]

        token_id = [self.get_token_id(c) for c in token]
        return token, token_id
    
    def decode(self, token_id, no_special=False):
        if no_special:
            for t in self.SPECIAL_VALUE:
                while t in token_id:
                    token_id.remove(t)
        token = [self.get_id_token(id) for id in token_id]
        return token, self.sc.join(token)

    
    def dict_size(self):
        return len(self.char_list)
    
class HuggingTokenizer(BaseTokenizer):
    def __init__(self, dict_path, sc='##'):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(dict_path)
        self.char_dict = self.tokenizer.get_vocab()
        self.char_list = list(self.char_dict.keys())
        self.sc = sc

    def get_token_id(self, token):
        return self.tokenizer.token_to_id(token.upper())

    def get_id_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def dict_size(self):
        return self.tokenizer.get_vocab_size()
    
    def encode(self, text, add_sos_eos=True):
        text = text.upper()
        output = self.tokenizer.encode(text)
        token, token_id = output.tokens, output.ids
        if add_sos_eos:
            token = [self.ID_KEY_SOS] + token + [self.ID_KEY_EOS]
            token_id = [self.ID_VALUE_SOS] + token_id + [self.ID_VALUE_SOS]
        return token, token_id

    def decode(self, token_id, no_special=False):
        if no_special:
            for t in self.SPECIAL_VALUE:
                while t in token_id:
                    token_id.remove(t)
        token = [self.get_id_token(id) for id in token_id]
        text = self.tokenizer.decode(token_id).replace(" " + self.sc, "")        
        return token, text

    @staticmethod
    def train_tokenizer(train_file, save_path, vocab_size=5000):
        tokenizer = Tokenizer(WordPiece(unk_token = BaseTokenizer.ID_KEY_UNK))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=BaseTokenizer.SPECIAL_KEY, vocab_size=vocab_size)
        tokenizer.train(files=train_file, trainer=trainer)
        tokenizer.save(save_path, pretty=True)

class SPMTokenizer(BaseTokenizer):
    def __init__(self, dict_path, sc='▁'):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(model_file=dict_path)
        self.char_dict = {}
        self.char_list = [i for i in range(self.tokenizer.vocab_size())]
        for k in [self.tokenizer.bos_id(), self.tokenizer.eos_id(),self.tokenizer.unk_id(),self.tokenizer.pad_id()]:
            if k in self.char_list:
                self.char_list.remove(k) 
        self.char_list = BaseTokenizer.SPECIAL_KEY + [self.tokenizer.id_to_piece(i) for i in self.char_list]

        for i,c in enumerate(self.char_list):
            self.char_dict[c] = i
        self.sc = sc

    def get_token_id(self, token):
        token = token.upper()
        if token in self.char_dict:
            return self.char_dict[token]
        else:
            return self.char_dict[self.ID_KEY_UNK]

    def get_id_token(self, id):
        if id > len(self.char_list):
            return self.ID_KEY_UNK
        else:
            return self.char_list[id]

    def dict_size(self):
        return len(self.char_list)

    def encode(self, text, add_sos_eos=True):
        text = text.upper()
        token = self.tokenizer.encode(text, out_type=str)
        token_id = [self.get_token_id(c) for c in token]
        if add_sos_eos:
            token = [self.ID_KEY_SOS] + token + [self.ID_KEY_EOS]
            token_id = [self.ID_VALUE_SOS] + token_id + [self.ID_VALUE_SOS]
        return token, token_id

    def decode(self, token_id, no_special=False):
        if no_special:
            for t in self.SPECIAL_VALUE:
                while t in token_id:
                    token_id.remove(t)
        token = [self.get_id_token(id) for id in token_id]
        text = ''.join([x.replace(self.sc, " ") for x in token])
        return token, text
    
    @staticmethod
    def train_tokenizer(train_file, save_path, vocab_size=5000):
        train_cmd = '--input={} --model_prefix={} --vocab_size={}'.format(train_file, save_path, vocab_size)
        spm.SentencePieceTrainer.train(train_cmd)