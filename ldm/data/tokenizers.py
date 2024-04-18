"""Copied and modified from ``https://github.com/openai/CLIP'',
``https://github.com/deep-floyd/IF''."""
import gzip
import html
from functools import lru_cache

import ftfy
import regex as re
import torch

__all__ = ['CLIPTokenizer']


@lru_cache()
def bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a
    large # of unicode characters in your vocab if you want to avoid UNKs. When
    you're at something like a 10B token dataset you end up needing around 5K
    for decent coverage. This is a signficant percentage of your normal, say,
    32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
    unicode strings. And avoids mapping to whitespace/control characters the
    bpe code barfs on.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length
    strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class CLIPTokenizer(object):

    def __init__(self,
                 length=77,
                 padding='zero',
                 bpe_path='tokenizers/clip/bpe_simple_vocab_16e6.txt.gz'):
        assert padding in ('zero', 'eos')
        self.length = length
        self.padding = padding
        self.bpe_path = bpe_path

        # init encoders and decoders
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = gzip.open(
            __file__.replace("ldm/data/tokenizers.py", "cache/bpe_simple_vocab_16e6.txt.gz")).read().decode(
            'utf-8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>'
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            # noqa: E501
            re.IGNORECASE)
        self.sos_token = self.encoder['<|startoftext|>']
        self.eos_token = self.encoder['<|endoftext|>']
        self.pad_token = 0 if padding == 'zero' else self.eos_token
        self.vocab_size = len(self.encoder)

    def __call__(self, sequence):
        return self.encode(sequence)

    def encode(self, sequence):
        if isinstance(sequence, str):
            return torch.LongTensor(self._encode(sequence))
        elif isinstance(sequence, (list, tuple)):
            return torch.LongTensor([self._encode(u) for u in sequence])
        else:
            raise TypeError(
                'Expected the "sequence" to be a string or a list, ' \
                f'but got {type(sequence)}'
            )

    def decode(self, tokens):
        if isinstance(tokens, torch.LongTensor):
            tokens = tokens.tolist()
        if isinstance(tokens[0], int):
            return self._decode(tokens)
        elif isinstance(tokens[0], list):
            return [self._decode(u) for u in tokens]
        else:
            raise TypeError(
                f'Expected the "tokens" to be a list of token IDs or a list of token '
                f'lists, but got a list of {type(tokens[0])}')

    def _encode(self, text):
        # get bpe-tokens
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self._bpe(token).split(' '))
        bpe_tokens = bpe_tokens[:self.length - 2]

        # append sos, eos, and pad tokens
        tokens = [self.sos_token] + bpe_tokens + [self.eos_token]
        tokens = tokens + [self.pad_token] * (self.length - len(tokens))
        return tokens

    def _decode(self, tokens):
        # remove sos, eos, and pad tokens
        tokens = [
            u for u in tokens
            if u not in (self.sos_token, self.eos_token, self.pad_token)
        ]

        # decode
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text
                          ]).decode('utf-8',
                                    errors='replace').replace('</w>', ' ')
        return text

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                    i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
