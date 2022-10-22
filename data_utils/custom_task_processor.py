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
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import csv
import os
from typing import List

import pandas as pd

from data_utils.task_processors import DataProcessor, PROCESSORS
from pet.utils import InputExample


# add by du jia.
class MRPCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    # Set this to the name of the task
    TASK_NAME = "mrpc"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["0", "1"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's task name
    TASK_NAME_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 3

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[MRPCProcessor.TEXT_A_COLUMN]
                text_b = line[MRPCProcessor.TEXT_B_COLUMN]
                task = line[MRPCProcessor.TASK_NAME_COLUMN]
                label = str(line[MRPCProcessor.LABEL_COLUMN])
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# add by du jia.
class QQPProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    # Set this to the name of the task
    TASK_NAME = "qqp"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["0", "1"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's task name
    TASK_NAME_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 3

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[QQPProcessor.TEXT_A_COLUMN]
                text_b = line[QQPProcessor.TEXT_B_COLUMN]
                task = line[QQPProcessor.TASK_NAME_COLUMN]
                label = str(line[QQPProcessor.LABEL_COLUMN])
            except IndexError:
                continue
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, task=task, label=label))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=i-1))
        return examples


#add by megaman
class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    # Set this to the name of the task
    TASK_NAME = "qnli"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.tsv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.tsv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.tsv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["entailment", "not_entailment"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's task name
    # TASK_NAME_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[QNLIProcessor.TEXT_A_COLUMN]
                text_b = line[QNLIProcessor.TEXT_B_COLUMN]
                task = "QNLI"
                label = str(line[QNLIProcessor.LABEL_COLUMN])
                if label == '':
                    if 'entailment' in text_b:
                        label = 'entailment'
                        text_b = text_b.strip()[:-10]
                    elif 'not_entailment' in text_b:
                        label = 'not_entailment'
                        text_b = text_b.strip()[:-14]
                    else:
                        # print('[DEBUG]===>',i, line)
                        ValueError('Invalid label.')
            except IndexError:
                # print('[DEBUG]===>',i, line)
                continue
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, task=task, label=label))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


#add by megaman
class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    # Set this to the name of the task
    TASK_NAME = "rte"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.tsv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.tsv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.tsv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["entailment", "not_entailment"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 0

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's task name
    # TASK_NAME_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[RTEProcessor.TEXT_A_COLUMN]
                text_b = line[RTEProcessor.TEXT_B_COLUMN]
                task = "QNLI"
                label = str(line[RTEProcessor.LABEL_COLUMN])
                if label == '':
                    if 'entailment' in text_b:
                        label = 'entailment'
                        text_b = text_b.strip()[:-10]
                    elif 'not_entailment' in text_b:
                        label = 'not_entailment'
                        text_b = text_b.strip()[:-14]
                    else:
                        print('[DEBUG]===>',i, line)
                        ValueError('Invalid label.')
            except IndexError:
                print('[DEBUG]===>',i, line)
                continue
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, task=task, label=label))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MNLIProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    """Processor for the MRPC data set (GLUE version)."""
    # Set this to the name of the task
    TASK_NAME = "mnli"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.tsv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev_matched.tsv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test_matched.tsv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.tsv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["contradiction", "entailment", "neutral"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 8

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 9

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = -1

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, MNLIProcessor.TRAIN_FILE_NAME)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, MNLIProcessor.DEV_FILE_NAME)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, MNLIProcessor.TEST_FILE_NAME)), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, MNLIProcessor.DEV_FILE_NAME)), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            idx = i
            try:
                text_a = line[MNLIProcessor.TEXT_A_COLUMN]
                text_b = line[MNLIProcessor.TEXT_B_COLUMN]
                task = 'mnli'
                label = str(line[MNLIProcessor.LABEL_COLUMN])
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=i))
        return examples


class SNLIProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    """Processor for the MRPC data set (GLUE version)."""
    # Set this to the name of the task
    TASK_NAME = "snli"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["contradiction", "entailment", "neutral"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 7

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 8

    # Set this to the column of the train/test csv files containing the input's task name
    TASK_NAME_COLUMN = -1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = -1

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            idx = i
            try:
                text_a = line[SNLIProcessor.TEXT_A_COLUMN]
                text_b = line[SNLIProcessor.TEXT_B_COLUMN]
                task = 'snli'
                label = str(line[SNLIProcessor.LABEL_COLUMN])
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=i))
        return examples


class MyTaskDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "my-task"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["1", "2", "3", "4"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 0

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TRAIN_FILE_NAME), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.DEV_FILE_NAME), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TEST_FILE_NAME), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MyTaskDataProcessor.LABELS

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                guid = "%s-%s" % (set_type, idx)
                label = row[MyTaskDataProcessor.LABEL_COLUMN]
                text_a = row[MyTaskDataProcessor.TEXT_A_COLUMN]
                text_b = row[MyTaskDataProcessor.TEXT_B_COLUMN] if MyTaskDataProcessor.TEXT_B_COLUMN >= 0 else None
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class SST2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    TASK_NAME="sst2"

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# register the processor for this task with its name
PROCESSORS[QQPProcessor.TASK_NAME] = QQPProcessor
PROCESSORS[MRPCProcessor.TASK_NAME] = MRPCProcessor
PROCESSORS[QNLIProcessor.TASK_NAME] = QNLIProcessor
PROCESSORS[RTEProcessor.TASK_NAME] = RTEProcessor
PROCESSORS[SNLIProcessor.TASK_NAME] = SNLIProcessor
PROCESSORS[MNLIProcessor.TASK_NAME] = MNLIProcessor
PROCESSORS[SST2Processor.TASK_NAME] = SST2Processor