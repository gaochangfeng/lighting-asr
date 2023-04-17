
import lasr.data.reader as reader

class Tokenizer(object):
    def __init__(self):
        self.ID_VALUE_BLACK = 0
        self.ID_VALUE_SOS = 1
        self.ID_VALUE_EOS = 2
        self.ID_VALUE_MASK = 3
        self.ID_VALUE_PAD = 4
        self.ID_VALUE_UNK = 5
        self.ID_VALUE_IGNORE = -1
        
        self.ID_KEY_BLACK = "<blank>"
        self.ID_KEY_SOS = "<sos>"
        self.ID_KEY_EOS = "<eos>"
        self.ID_KEY_MASK = "[MASK]"
        self.ID_KEY_PAD = "[PAD]"
        self.ID_KEY_UNK = "[UNK]"

        self.SPECIAL_VALUE = [
            self.ID_VALUE_BLACK,
            self.ID_VALUE_SOS,
            self.ID_VALUE_EOS,
            self.ID_VALUE_MASK,
            self.ID_VALUE_PAD,
            self.ID_VALUE_UNK,            
        ]


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

class CharTokenizer(Tokenizer):
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