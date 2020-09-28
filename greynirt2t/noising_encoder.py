"""

    Noising SubwordTextEncoder

    Copyright (C) 2020 Miðeind ehf.
    Original author: Haukur Barri Símonarson

    This software is licensed under the MIT License:
        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Noising SubwordTextEncoder is a SubwordTextEncoder implementation that
    adds word noise as described in https://arxiv.org/abs/1808.09381
"""

import numpy as np
from operator import itemgetter


from tensor2tensor.data_generators.text_encoder import (
    SubwordTextEncoder,
    tokenizer as t2t_tokenizer,
    RESERVED_TOKENS as T2T_RESERVED_TOKENS,
    native_to_unicode,
)

import tokenizer as greynir_tokenizer

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


UNK = "<unk>"
RESERVED_TOKENS = T2T_RESERVED_TOKENS + [UNK]
UNK_ID = RESERVED_TOKENS.index(UNK)

def shuffle_noise(item_list, k=3, p=0.10):
    displacements = np.random.uniform(0, k, len(item_list))
    idxs = [
        (idx_clean, idx_clean + displacement)
        for (idx_clean, displacement) in enumerate(displacements)
    ]
    idxs = sorted(idxs, key=itemgetter(1))
    ic(idxs)
    ret = []
    for (idx_clean, _noised_idx) in idxs:
        ret.append(item_list[idx_clean])
    return ret


def blank_noise(item_list, blank_token, p=0.10):
    keep_probs = np.random.random(len(item_list)) > p
    ic(keep_probs)
    ret = [item if keep else blank_token for (item, keep) in zip(item_list, keep_probs)]
    return ret


def drop_noise(item_list, p=0.10):
    keep_probs = np.random.random(len(item_list)) > p
    ic(keep_probs)
    ret = [item for (item, keep) in zip(item_list, keep_probs) if keep]
    return ret


class NoisingSubwordTextEncoder(SubwordTextEncoder):
    def __init__(
        self,
        filename=None,
        prob_dropout=0.1,
        prob_blank=0.1,
        prob_shuffle=0.1,
        max_shuffle_dist=3,
    ):
        self.prob_dropout = prob_dropout
        self.prob_blank = prob_blank
        self.prob_shuffle = prob_shuffle
        self.max_shuffle_dist = max_shuffle_dist
        super(NoisingSubwordTextEncoder, self).__init__(filename=filename)

    def encode_with_noise(self, string):
        greynir_tokens = [tok for tok in greynir_tokenizer.tokenize(string) if tok.txt]
        if self.prob_shuffle:
            greynir_tokens = drop_noise(greynir_tokens, p=self.prob_dropout)
        if self.prob_shuffle:
            greynir_tokens = shuffle_noise(greynir_tokens, p=self.prob_shuffle)
        text = greynir_tokenizer.detokenize(greynir_tokens)
        t2t_tokens = t2t_tokenizer.encode(text)
        if self.prob_blank:
            t2t_tokens = blank_noise(t2t_tokens, UNK, p=self.prob_blank)
        return self._tokens_to_subtoken_ids(t2t_tokens)

    def _load_from_file(self, path):
        with open(path, "r") as fp:
            subtoken_strings = []
            for line in fp:
                s = line.rstrip()
                # Some vocab files wrap words in single quotes, but others don't
                if (s.startswith("'") and s.endswith("'")) or (
                    s.startswith('"') and s.endswith('"')
                ):
                    s = s[1:-1]
                subtoken_strings.append(s)
        self._init_subtokens_from_list(subtoken_strings, reserved_tokens=RESERVED_TOKENS)
        self._init_alphabet_from_tokens(subtoken_strings)


    def _tokens_to_subtoken_ids(self, tokens):
        ret = []
        for token in tokens:
            if token is UNK:
                ret.extend([UNK_ID])
                continue
            ret.extend(self._token_to_subtoken_ids(token))
        return ret
