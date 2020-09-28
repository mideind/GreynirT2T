"""

    T2T problem descriptions for translation augmented with
    backtranslation from Icelandic to English

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

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import functools
import itertools

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.models import lstm
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import registry

import tensorflow as tf
import numpy as np


from . import translate_enis
from . import noising_encoder

add_task_id = translate_enis.add_task_id
TranslateEnis16kV4 = translate_enis.TranslateEnis16kV4
TranslateEnis16k = translate_enis.TranslateEnis16k
PartialTargetsTranslateProblem = translate_enis.PartialTargetsTranslateProblem
NoisingSubwordTextEncoder = noising_encoder.NoisingSubwordTextEncoder

EOS = text_encoder.EOS_ID

FLAGS = tf.flags.FLAGS

@registry.register_problem
class NewscrawlBtIsen16NoisedkTagged(PartialTargetsTranslateProblem):

    @property
    def is_generate_per_split(self):
        # dataset is not pre-split
        return False

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        raise NotImplemented

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        noising_enc = NoisingSubwordTextEncoder(os.path.join(data_dir, TranslateEnis16k.vocab_filename))
        encoders = {
            "inputs": noising_enc,
            "targets": encoder,
        }
        return encoders

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_tab_iterator):
        if dataset_split == problem.DatasetSplit.TRAIN:
            iter_ = custom_iterator("/users/home/hbs24/data/newscrawl-bt.is-en.tsv")
            return iter_

    def dataset(self,
                mode,
                data_dir=None,
                num_threads=None,
                output_buffer_size=None,
                shuffle_files=None,
                hparams=None,
                preprocess=True,
                dataset_split=None,
                shard=None,
                partition_id=0,
                num_partitions=1,
                shuffle_buffer_size=1024,
                max_records=-1):
        stream = super(NewscrawlBtIsen16NoisedkTagged, self).dataset(
                mode,
                data_dir=data_dir,
                num_threads=num_threads,
                output_buffer_size=output_buffer_size,
                shuffle_files=shuffle_files,
                hparams=hparams,
                preprocess=preprocess,
                dataset_split=dataset_split,
                shard=shard,
                partition_id=partition_id,
                num_partitions=num_partitions,
                shuffle_buffer_size=shuffle_buffer_size,
                max_records=max_records,
                )
        if self.task_id == -1:
            raise ValueError("Must provide task_id")
        stream = stream.map(lambda example: add_task_id(self.task_id, example))
        if mode == tf.estimator.ModeKeys.TRAIN:
            stream = stream.repeat()
        return stream


@registry.register_problem
class TranslateIsen16kV4TaggedBtNoisedMix3(PartialTargetsTranslateProblem):

    mixing_weights = [33,67]
    other_problem = NewscrawlBtIsen16NoisedkTagged

    @property
    def is_generate_per_split(self):
        # dataset is not pre-split
        return False

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        raise NotImplemented

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_tab_iterator):
        return iter([])

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        enis = TranslateEnis16kV4()
        enis.generate_data(data_dir, tmp_dir, task_id=task_id)
        other_problem = self.other_problem()
        other_problem.generate_data(data_dir, tmp_dir, task_id=task_id)

    def dataset(self,
                mode,
                data_dir=None,
                num_threads=None,
                output_buffer_size=None,
                shuffle_files=None,
                hparams=None,
                preprocess=True,
                dataset_split=None,
                shard=None,
                partition_id=0,
                num_partitions=1,
                shuffle_buffer_size=1024,
                max_records=-1):
        enis = TranslateEnis16kV4(was_reversed=True).dataset(
                mode,
                data_dir=data_dir,
                num_threads=num_threads,
                output_buffer_size=output_buffer_size,
                shuffle_files=shuffle_files,
                hparams=hparams,
                preprocess=preprocess,
                dataset_split=dataset_split,
                shard=shard,
                partition_id=partition_id,
                num_partitions=num_partitions,
                shuffle_buffer_size=shuffle_buffer_size,
                max_records=max_records,
                )
        other_problem = self.other_problem()
        other_problem = other_problem.dataset(
                mode,
                data_dir=data_dir,
                num_threads=num_threads,
                output_buffer_size=output_buffer_size,
                shuffle_files=shuffle_files,
                hparams=hparams,
                preprocess=preprocess,
                dataset_split=dataset_split,
                shard=shard,
                partition_id=partition_id,
                num_partitions=num_partitions,
                shuffle_buffer_size=shuffle_buffer_size,
                max_records=max_records,
                )

        datasets = []
        for dataset in [enis, other_problem]:
            if is_training:
                dataset = task_dataset.repeat()

            datasets.append(task_dataset)

        sampled_dataset = tf.data.experimental.sample_from_datasets(
            datasets,
            weights=np.asarray(self.mixing_weights, dtype=np.float64))
        return sampled_dataset

    def input_fn(self,
            mode,
            hparams,
            data_dir=None,
            params=None,
            config=None,
            force_repeat=False,
            prevent_repeat=False,
            dataset_kwargs=None):

        prob_hp = self.get_hparams()
        data_dir = data_dir or hparams.data_dir
        real_vocab_size = prob_hp.vocabulary["inputs"].vocab_size
        if self.feature_encoders(data_dir)["inputs"].vocab_size == real_vocab_size:
            self._hparams.vocab_size["inputs"] = real_vocab_size + 1
            self._hparams.vocab_size["targets"] = real_vocab_size + 1
        task_id = self._hparams.vocab_size["inputs"] - 1
        enis = TranslateEnis16kV4(was_reversed=True)
        enis._hparams = self._hparams
        enis_stream = enis.input_fn(
                mode,
                hparams,
                data_dir=data_dir,
                params=params,
                config=config,
                force_repeat=force_repeat,
                prevent_repeat=prevent_repeat,
                dataset_kwargs=dataset_kwargs,
                )
        if mode == tf.estimator.ModeKeys.PREDICT:
            return enis_stream

        other_problem = self.other_problem()
        other_problem._hparams = self._hparams
        other_problem._task_id = task_id
        other_problem_stream = other_problem.input_fn(
                mode,
                hparams,
                data_dir=data_dir,
                params=params,
                config=config,
                force_repeat=force_repeat,
                prevent_repeat=prevent_repeat,
                dataset_kwargs=dataset_kwargs,
                )
        streams = [enis_stream, other_problem_stream]

        sampled_dataset = tf.data.experimental.sample_from_datasets(
            streams,
            weights=np.asarray(self.mixing_weights, dtype=np.float64))
        return sampled_dataset

    def source_data_files(self, dataset_split):
        return []

    def get_hparams(self, model_hparams=None):
        prob_hp = super(TranslateIsen16kV4TaggedBtNoisedMix3, self).get_hparams(model_hparams=model_hparams)
        vocab_size = prob_hp.vocabulary["inputs"].vocab_size
        vocab_size_with_task_id = 16421 + 1
        if vocab_size != vocab_size_with_task_id:
            self._hparams.vocab_size["inputs"] = vocab_size_with_task_id
            self._hparams.vocab_size["targets"] = vocab_size_with_task_id
        return prob_hp
