# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The PVP is responsible for applying patterns to inputs and mapping
labels to their verbalizations (see the paper for more details on PVPs).
This file shows an example of a PVP for a new task.
"""

import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict
import log

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer

from pet.utils import InputExample, get_verbalization_ids

logger = log.get_logger('root')

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, pattern_id: int = 0, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        """
        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)
        """

        ## if self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER, wrp.PLM_WRAPPER]:
        self.template = self._load_template(pattern_id)
        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer

        parts_a, parts_b, block_flag_a, block_flag_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        # self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)
        num_special = self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length - num_special)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        # tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else []

        # add
        assert len(parts_a) == len(block_flag_a)
        assert len(parts_b) == len(block_flag_b)

        block_flag_a = [flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part]
        block_flag_b = [flag for (part, _), flag in zip(parts_b, block_flag_b) for _ in part]

        assert len(tokens_a) == len(block_flag_a)
        assert len(tokens_b) == len(block_flag_b)

        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a, block_flag_b)
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)


        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        assert len(input_ids) == len(block_flag)

        # return input_ids, token_type_ids
        return input_ids, token_type_ids, block_flag


    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)


    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one [MASK] per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize

    @abstractmethod
    def _load_template(self, pattern_id) -> str:
        pass

    # add by du jia
    def template_filler(self, text_a, text_b):
        placeholder = {token: token for ids, token in enumerate(set(self.template))}
        if '<soft>' in placeholder:
            placeholder['<soft>'] = '*'
        placeholder['<text_a>'] = text_a
        if '<text_b>' in placeholder:
            placeholder['<text_b>'] = text_b
        if '[MASK]' in placeholder:
            placeholder['[MASK]'] = self.mask
        if '<mask>' in placeholder:
            placeholder['<mask>'] = self.mask

        string_list_a = [placeholder[token] for token in self.template]
        string_list_b = []
        block_flag_a = [1 if token == '<soft>' else 0 for token in self.template]
        block_flag_b = []
        # print(string_list_a, string_list_b)
        assert len(string_list_a) == len(block_flag_a)
        assert len(string_list_b) == len(block_flag_b)
        return string_list_a, string_list_b, block_flag_a, block_flag_b


class QNLIPVP(PVP):
    # VERBALIZER = {
    #     "0": ["Alas"],
    #     "1": ["Rather"]
    # }
    VERBALIZER = {
        "entailment": ["Recently"],
        "not_entailment": ["Fortunately"]
    }
    TEMPLATES = {
      "0" : "<text_a> ? [MASK] , it's true , <text_b> .",
      "1" : "<text_a> <soft> [MASK] , but <text_b> .",
      "2" : "<text_a> <soft> [MASK] <soft> but <text_b> .",
      "3" : "<text_a> <soft> [MASK] <soft> but <text_b> <soft>",
      "4" : "<soft> <text_a> <soft> <text_b> <soft> [MASK] .",
      "5" : "<soft> <text_a> <soft> <soft> <text_b> <soft> [MASK] .",
      "6" : "<soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] .",
      "7" : "<soft> <soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] ."
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return QNLIPVP.VERBALIZER[label]

    # 从文件中加载会比较慢，所以改成类变量的形式
    def _load_template(self, pattern_id):
        templates = QNLIPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template

    # def _load_template_from_file(self, pattern_dir: str, pattern_id):
    #     pattern_file = os.path.join(pattern_dir, 'qqp.json')
    #     pattern_id = str(pattern_id)
    #     with open(pattern_file, 'r') as f:
    #         templates = json.load(f)
    #     assert pattern_id in templates
    #     template = templates[str(pattern_id)].strip().split()
    #     return template, pattern_file


class QQPPVP(PVP):
    # VERBALIZER = {
    #     "0": ["Alas"],
    #     "1": ["Rather"]
    # }
    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }

    TEMPLATES = {
      "0" : "<text_a> ? [MASK] , but <text_b> .",
      "1" : "<text_a> <soft> [MASK] , but <text_b> .",
      "2" : "<text_a> <soft> [MASK] <soft> but <text_b> .",
      "3" : "<text_a> ? [MASK] <soft> <soft> <soft> <text_b>",  # 所有soft token都在MASK之后, 保留anchor
      "4" : "<soft> <text_a> <soft> <text_b> <soft> [MASK] .",
      "5" : "<soft> <text_a> <soft> <soft> <text_b> <soft> [MASK] .",
      "6" : "<soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] .",
      "7" : "<soft> <soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] ."
    }

    # TEMPLATES = {
    #   "0" : "<text_a> ? [MASK] , but <text_b> .",
    #   "1" : "<text_a> <soft> [MASK] , but <text_b> .",
    #   "2" : "<text_a> <soft> [MASK] <soft> but <text_b> .",
    #   "3" : "<text_a> <soft> [MASK] <soft> but <text_b> <soft>",
    #   "4" : "<soft> <text_a> <soft> <text_b> <soft> [MASK] .",
    #   "5" : "<soft> <text_a> <soft> <soft> <text_b> <soft> [MASK] .",
    #   "6" : "<soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] .",
    #   "7" : "<soft> <soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] ."
    # }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return QQPPVP.VERBALIZER[label]

    # 从文件中加载会比较慢，所以改成类变量的形式
    def _load_template(self, pattern_id):
        templates = QQPPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template

    # def _load_template_from_file(self, pattern_dir: str, pattern_id):
    #     pattern_file = os.path.join(pattern_dir, 'qqp.json')
    #     pattern_id = str(pattern_id)
    #     with open(pattern_file, 'r') as f:
    #         templates = json.load(f)
    #     assert pattern_id in templates
    #     template = templates[str(pattern_id)].strip().split()
    #     return template, pattern_file


class MRPCPVP(PVP):
    # VERBALIZER = {
    #     "0": ["Alas"],
    #     "1": ["Rather"]
    # }

    VERBALIZER = {
        "0": ["Thus"],
        "1": ["At"]
    }

    # TEMPLATES = {
    #   "0": "<text_a> . [MASK] However <text_b> .",
    #   "1" : "<text_a> <soft> [MASK] However <text_b> .",
    #   "2" : "<text_a> <soft> [MASK] <soft> <text_b> .",
    #   "3" : "<text_a> <soft> [MASK] <soft> <text_b> <soft>",
    #   "4" : "<soft> <text_a> <soft> <text_b> <soft> [MASK] <soft>",
    # #   "4" : "<soft> <text_a> <soft> <text_b> <soft> [MASK] .",
    #   "5" : "<soft> <text_a> <soft> <soft> <text_b> <soft> [MASK] .",
    #   "6" : "<soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] .",
    #   "7" : "<soft> <soft> <text_a> <soft> <soft> <text_b> <soft> <soft> [MASK] ."
    # }

    TEMPLATES = {  # infix prompt
    #   "0": "<text_a> . [MASK] . In fact , <text_b>",  # 06.14 case anlysis 1
    #   "0": "<text_a> . [MASK] . This is the first time <text_b>",  # 06.14 case anlysis 2
    #   "0": "<text_a> . [MASK] , <text_b>",   # 06.14 case anlysis 3
      "0": "<text_a> . [MASK] However <text_b> .",   # 06.14 case anlysis 4
      "1" : "<text_a> . [MASK] <soft> <text_b>",
      "2" : "<text_a> <soft> [MASK] <soft> <text_b> .",
    #   "3" : "<text_a> . [MASK] <soft> <soft> <soft> <text_b>",  # original
      "3" : "<soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
    #   "6" : "<soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
    #   "9" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
      "12" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
      "15" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
      "18" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
      "21" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <text_b> [MASK] ",  # 前缀方式处理
      "4" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <text_b>",
      "5" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
      "6" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "7" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "8" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "9" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "10" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "15" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "20" : "<text_a> . [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    }

    # TEMPLATES = {  # infix prompt
    #   "0": "<text_a> ? [MASK] , <text_b>",
    #   "1" : "<text_a> ? [MASK] <soft> <text_b>",
    #   "2" : "<text_a> ? [MASK] <soft> <soft> <text_b> .",
    #   "3" : "<text_a> ? [MASK] <soft> <soft> <soft> <text_b>",
    #   "4" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <text_b>",
    #   "5" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "6" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "7" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "8" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "9" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "10" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "15" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "20" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    # }

    # TEMPLATES = {  # infix prompt
    #   "0": "<text_a> ? <mask> , <text_b>",
    #   "1" : "<text_a> ? <mask> <soft> <text_b>",
    #   "2" : "<text_a> <mask> <soft> <soft> <text_b> .",
    #   "3" : "<text_a> <mask> <soft> <soft> <soft> <text_b>",
    #   "4" : "<text_a> <mask> <soft> <soft> <soft> <soft> <text_b>",
    #   "5" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "6" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "7" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "8" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "9" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "10" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "15" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "20" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    # }
    # TEMPLATES = {  # prefix prompt
    #   "0": "<text_a> ? <mask> , <text_b>",
    #   "1" : "<soft> <text_a> ? <mask> <text_b>",
    #   "2" : "<soft> <soft> <text_a> <mask> <text_b> .",
    #   "3" : "<soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "4" : "<soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "5" : "<soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "6" : "<soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "7" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "8" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "9" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "10" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "15" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "20" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    # }
    # TEMPLATES = {  # suffix prompt
    #   "0": "<text_a> ? <mask> , <text_b>",
    #   "1" : "<soft> <text_a> ? <mask> <text_b>",
    #   "2" : "<text_a> <mask> <text_b> . <soft> <soft>",
    #   "3" : "<text_a> <mask> <text_b> <soft> <soft> <soft>",
    #   "4" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft>",
    #   "5" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft> <soft>",
    #   "6" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft> <soft> <soft>",
    #   "7" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft> <soft> <soft> <soft>",
    #   "8" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>",
    #   "9" : "<text_a> <mask> <text_b> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>",
    #   "10" : "<soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <mask> <text_b>",
    #   "15" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "20" : "<text_a> <mask> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    # }
    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return MRPCPVP.VERBALIZER[label]

    # 从文件中加载模版较慢，因此改成通过类变量加载
    def _load_template(self, pattern_id):
        templates = MRPCPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template

    # def _load_template_from_file(self, pattern_dir: str, pattern_id):
    #     pattern_file = os.path.join(pattern_dir, 'mrpc.json')
    #     pattern_id = str(pattern_id)
    #     with open(pattern_file, 'r') as f:
    #         templates = json.load(f)
    #     assert pattern_id in templates
    #     template = templates[str(pattern_id)].strip().split()
    #     return template, pattern_file


class MyTaskPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "my-task"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech"]
    }

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.
        """

        # We tell the tokenizer that both text_a and text_b can be truncated if the resulting sequence is longer than
        # our language model's max sequence length.
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id == 0:
            # this corresponds to the pattern [MASK]: a b
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            # this corresponds to the pattern [MASK] News: a || (b)
            return [self.mask, 'News:', text_a], ['(', text_b, ')']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return MyTaskPVP.VERBALIZER[label]


class SNLIPVP(PVP):
    VERBALIZER = {
        "contradiction": ["But"],
        "entailment": ["YES"],
        "neutral": ["This"],
    }

    TEMPLATES = {  # infix prompt
      "0": "<text_a> ? [MASK] , <text_b>",
      "1" : "<text_a> ? [MASK] <soft> <text_b>",
      "2" : "<text_a> [MASK] <soft> <soft> <text_b> .",
      "3" : "<text_a> [MASK] <soft> <soft> <soft> <text_b>",
      "4" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <text_b>",
      "5" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
      "6" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "7" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "8" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "9" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "10" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "15" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "20" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return SNLIPVP.VERBALIZER[label]

    # 从文件中加载模版较慢，因此改成通过类变量加载
    def _load_template(self, pattern_id):
        templates = SNLIPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template


class MNLIPVP(PVP):
    VERBALIZER = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"],
    }

    TEMPLATES = {  # infix prompt
      "0": "<text_a> ? [MASK] , <text_b>",
      "1" : "<text_a> ? [MASK] <soft> <text_b>",
      "2" : "<text_a> ? [MASK] <soft> <soft> <text_b> .",
      "3" : "<text_a> ? [MASK] <soft> <soft> <soft> <text_b>",
      "4" : "<text_a> ? [MASK] <soft> <soft> <soft> <soft> <text_b>",
      "5" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
      "6" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "7" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "8" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "9" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "10" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "15" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "20" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return MNLIPVP.VERBALIZER[label]

    # 从文件中加载模版较慢，因此改成通过类变量加载
    def _load_template(self, pattern_id):
        templates = MNLIPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template


class RTEPVP(PVP):
    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    # TEMPLATES = {  # infix prompt
    #   "0": "<text_a> ? [MASK] , <text_b>",
    #   "1" : "<text_a> [MASK] <soft> <text_b>",
    #   "2" : "<text_a> [MASK] <soft> <soft> <text_b> .",
    #   "3" : "<text_a> [MASK] <soft> <soft> <soft> <text_b>",
    #   "4" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <text_b>",
    #   "5" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "6" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "7" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "8" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "9" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "10" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "15" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
    #   "20" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    # }

    TEMPLATES = {  # infix prompt
      "0": "<text_a> Question: <text_b> ? the Answer: [MASK] .",
      "1" : "<text_a> Question: <text_b> ? <soft> Answer: [MASK] .",
      "2" : "<text_a> [MASK] <soft> <soft> <text_b> .",
      "3" : "<text_a> [MASK] <soft> <soft> <soft> <text_b>",
      "4" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <text_b>",
      "5" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <text_b>",
      "6" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "7" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "8" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "9" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "10" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "15" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_b>",
      "20" : "<text_a> [MASK] <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft>  <text_b>",
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        return self.template_filler(text_a, text_b)

    def verbalize(self, label) -> List[str]:
        return RTEPVP.VERBALIZER[label]

    # 从文件中加载会比较慢，所以改成类变量的形式
    def _load_template(self, pattern_id):
        templates = RTEPVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template


class SST2PVP(PVP):

    VERBALIZER = {
        "0": ["bad"],
        "1": ["wonderful"]
    }

    TEMPLATES = {
      "0": "<text_a> It was <mask> .",
      "1": "<text_a> <soft> was <mask> .",
      "2": "<text_a> <soft> <soft> <mask> .",
      "3": "<text_a> <soft> <soft> <soft> <mask> .",
      "4": "<text_a> <soft> <soft> <soft> <soft> <mask> .",
      "6": "<text_a> <soft> <soft> <soft> <soft> <soft> <soft> <mask> .",
      "8": "<text_a> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <mask> .",
      "16": "<text_a> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <text_a> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <soft> <mask> .",
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)

        return self.template_filler(text_a, text_b=None)

    def verbalize(self, label) -> List[str]:
        return SST2PVP.VERBALIZER[label]

    # 从文件中加载模版较慢，因此改成通过类变量加载
    def _load_template(self, pattern_id):
        templates = SST2PVP.TEMPLATES
        assert str(pattern_id) in templates
        template = templates[str(pattern_id)].strip().split()
        return template

# register the PVP for this task with its name
PVPS = {
    'mrpc': MRPCPVP,
    'qqp': QQPPVP,
    'qnli': QNLIPVP,
    'snli': SNLIPVP,
    'mnli': MNLIPVP,
    'rte': RTEPVP,
    'sst2': SST2PVP
}
