"""

    T2T problem descriptions for translation between English-Icelandic

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

EOS = text_encoder.EOS_ID

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool("scoring", False,
        "Whether to use provided targets and score them instead of inference from input_ids in predict-mode.",
)



def tabbed_generator_samples(source_path):
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        for line in source_file:
            yield line


def line_stream_to_field_stream(lines, fields=None):
    lines = iter(lines)
    for line in lines:
        for idx, field in enumerate(line.split("\n")):
            if fields is not None and idx in fields:
                yield field
            else:
                yield field


def add_task_id(task_id, example):
    concat_list = [[task_id], example['inputs']]
    example['inputs'] = tf.concat(concat_list, axis=0)
    return example


class TranslateProblemV2(translate.TranslateProblem):
    def __init__(self, was_reversed=False, was_copy=False):
        super(TranslateProblemV2, self).__init__(was_reversed, was_copy)

    def generate_encoded_samples(self, data_dir, tmp_dir, train):
        vocabs = self.feature_encoders(data_dir)
        source_vocab = vocabs["inputs"]
        target_vocab = vocabs["targets"]

        eos_list = [EOS]
        for sample in self.generate_samples(data_dir, tmp_dir, train):
            source_vocab = vocabs["inputs"]
            target_vocab = vocabs["targets"]
            yield {
                "inputs": source_vocab.encode(sample["inputs"]) + eos_list,
                "targets": target_vocab.encode(sample["targets"]) + eos_list,
            }


class ScorableTranslateProblem(translate.TranslateProblem):

    def serving_input_fn(self, hparams, decode_hparams=None, use_tpu=False):
        """Input fn for serving export, when using TransformerScorer, starting from serialized example."""
        # This override is a work around when exporting using transformerscorer
        mode = tf.estimator.ModeKeys.PREDICT
        serialized_example = tf.placeholder(
            dtype=tf.string, shape=[None], name="serialized_example")
        dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
        dataset = dataset.map(self.decode_example)
        dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
        dataset = dataset.map(data_reader.cast_ints_to_int32)

        if use_tpu:
          padded_shapes = data_reader.pad_for_tpu(dataset.output_shapes, hparams,
                                                  hparams.max_length)
          batch_size = 1 if not decode_hparams else getattr(decode_hparams,
                                                            "batch_size", 1)
          dataset = dataset.padded_batch(
              batch_size, padded_shapes, drop_remainder=False)
          dataset = dataset.map(
              functools.partial(data_reader.pad_batch, batch_multiple=batch_size))
        else:
          dataset = dataset.padded_batch(
              tf.shape(serialized_example, out_type=tf.int64)[0],
              dataset.output_shapes)

        dataset = dataset.map(data_reader.standardize_shapes)
        features = tf.data.experimental.get_single_element(dataset)

        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=serialized_example)


class PartialTargetsTranslateProblem(ScorableTranslateProblem):
    def example_reading_spec(self):
        """Define how data is serialized to file and read back.

        Returns:
            data_fields: A dictionary mapping data names to its feature type.
            data_items_to_decoders: A dictionary mapping data names to TF Example
                decoders, to be used when reading back TF examples from disk.
        """
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "partial_targets": tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)


@registry.register_problem
class TranslateEnis16k(ScorableTranslateProblem):

    @property
    def source_vocab_size(self):
        return 2 ** 14  # 16384

    @property
    def vocab_type(self):
        return text_problems.VocabType.SUBWORD

    @property
    def targeted_vocab_size(self):
        return self.source_vocab_size

    @property
    def approx_vocab_size(self):
        return self.source_vocab_size

    @property
    def is_generate_per_split(self):
        # dataset is unified, let tf split for us
        return False

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        source_filename = "en-is.sample_for_vocab.tsv"
        source_path = os.path.join(tmp_dir, source_filename)
        return tabbed_generator_samples(source_path)

    def source_data_files(self, dataset_split):
        return [["en-is-v2.tar.gz", ("en-is.en", "en-is.is")]]


@registry.register_problem
class TranslateEnis16kV3(ScorableTranslateProblem):

    @property
    def is_generate_per_split(self):
        # dataset is pre-split
        return True

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        raise NotImplemented

    def source_dataset_files(self, data_dir, tmp_dir):
        raise NotImplemented

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_iterator):
        if dataset_split == problem.DatasetSplit.TRAIN:
            return custom_iterator("/data/en-is/v3/train/en-is.train.en", "/data/en-is/v3/train/en-is.train.is")
        return custom_iterator("/data/en-is/v3/dev/en-is.dev.en", "/data/en-is/v3/dev/en-is.dev.is")


@registry.register_problem
class TranslateEnis16kV4(PartialTargetsTranslateProblem):

    @property
    def is_generate_per_split(self):
        # dataset is pre-split
        return True

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        raise NotImplemented

    def source_dataset_files(self, data_dir, tmp_dir):
        raise NotImplemented

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_iterator):
        if dataset_split == problem.DatasetSplit.TRAIN:
            return custom_iterator("/data/en-is/v4/train/en-is.train.en", "/data/en-is/v4/train/en-is.train.is")
        return custom_iterator("/data/en-is/v4/dev/en-is.dev.bleu_sample.en", "/data/en-is/v4/dev/en-is.dev.bleu_sample.is")


@registry.register_problem
class TranslateEnis16kV4Bpe(ScorableTranslateProblem):

    @property
    def is_generate_per_split(self):
        # dataset is pre-split
        return True

    @property
    def batch_size_means_tokens(self):
        return True

    def source_dataset_files(self, data_dir, tmp_dir):
        raise NotImplemented

    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**14  # 16384

    @property
    def vocab_type(self):
        return "bpe"

    @property
    def vocab_filename(self):
        other_problem = self.use_vocab_from_other_problem
        if other_problem:
            return other_problem.vocab_filename
        return "vocab.%s.%d.%s" % (self.dataset_filename(),
                self.approx_vocab_size,
                self.vocab_type)

    @property
    def skip_random_fraction_when_training(self):
      # the custom self.dataset function already skips random fraction when training,
      # therefore we prevent an extra skip in data_reader.input_fn
      return False

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        try:
            path = os.path.join(data_dir, self.vocab_filename)
            enc = bpe.BPEEncoder.load_from_file(path)
            return enc
        except FileNotFoundError as e:
            if force_get:
                raise e from None
        return self._create_vocab(data_dir, tmp_dir)

    def _create_vocab(self, data_dir, tmp_dir):
        line_budget = int(1e5)

        def example_dict_to_fields(dict_stream):
            for d in dict_stream:
                yield d["inputs"]
                yield d["targets"]

        samples = self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)
        samples = example_dict_to_fields(samples)
        max_lines = 200000
        enc = bpe.BPEEncoder.build_from_generator(samples, self.approx_vocab_size, max_lines=max_lines)
        enc.store_to_file(os.path.join(data_dir, self.vocab_filename))
        return enc

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_iterator):
        if dataset_split == problem.DatasetSplit.TRAIN:
            return custom_iterator("/data/en-is/v4/train/en-is.train.en", "/data/en-is/v4/train/en-is.train.is")
        return custom_iterator("/data/en-is/v4/dev/en-is.dev.bleu_sample.en", "/data/en-is/v4/dev/en-is.dev.bleu_sample.is")

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
        }

        all_paths = []
        split_paths = []
        for dataset_split in self.dataset_splits:
            split_tag = dataset_split["split"]  # train/dev/test
            shards = dataset_split["shards"]
            paths = filepath_fns[split_tag](data_dir, shards, shuffled=self.already_shuffled)
            all_paths.extend(paths)
            split_paths.append((split_tag, paths))

        if self.is_generate_per_split:
            for split_tag, paths in split_paths:
                stream = self.generate_samples(data_dir, tmp_dir, split_tag)
                data_utils.write_stream_to_files(stream, paths)
                data_utils.shuffle_many(paths)
        else:
            stream = self.generate_samples(data_dir, tmp_dir, split_tag)
            data_utils.write_stream_to_files(stream, all_paths)
            data_utils.shuffle_many(all_paths, transform=self._pack_fn())

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
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            shuffle_files = shuffle_files or shuffle_files is None and is_training

            dataset_split = dataset_split or mode
            assert data_dir

            if hparams is None:
                hparams = problem.default_model_hparams()

            if not hasattr(hparams, "data_dir"):
                hparams.add_hparam("data_dir", data_dir)
            if not hparams.data_dir:
                hparams.data_dir = data_dir
            # Construct the Problem's hparams so that items within it are accessible
            _ = self.get_hparams(hparams)

            data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
            tf.logging.info("Reading data files from %s", data_filepattern)

            stream = data_utils.read_many(data_filepattern, skip_random_fraction=is_training, repeat=is_training)

            encoder = TranslateEnis16k().get_or_create_vocab(data_dir, data_dir, force_get=True)

            def line_to_ex(line):
                src, tgt = line.split("\t")
                if is_training:
                    inputs = encoder.encode(src)
                    targets = encoder.encode(tgt)
                else:
                    inputs = encoder.encode(src)
                    targets = encoder.encode(tgt)
                dd =  dict(
                    inputs=inputs + [EOS],
                    targets=targets + [EOS]
                )
                return dd

            ex_shapes = {"inputs":tf.TensorShape([None,]), "targets":tf.TensorShape([None,])}
            ex_types = {"inputs":tf.int32, "targets":tf.int32}
            if is_training:
                stream = data_utils.bucket_shuffle_iterable(stream, shuffle_buffer_size)
            stream = map(line_to_ex, stream)

            dataset = tf.data.Dataset.from_generator(lambda: stream, ex_types, output_shapes=ex_shapes)

            if preprocess:
                dataset = self.preprocess(dataset, mode, hparams,
                                        interleave=shuffle_files)
            return dataset


@registry.register_problem
class TranslateEnis16kV4BpeBaked(ScorableTranslateProblem):

    @property
    def is_generate_per_split(self):
        # dataset is pre-split
        return True

    @property
    def batch_size_means_tokens(self):
        return True

    def source_dataset_files(self, data_dir, tmp_dir):
        raise NotImplemented

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16kV4Bpe().get_or_create_vocab(data_dir, tmp_dir, force_get=True)

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        vocabs = self.feature_encoders(data_dir)
        eos_list = [EOS]
        enc = vocabs["inputs"]
        done = False

        if dataset_split == problem.DatasetSplit.TRAIN:
            for inputs, targets in bpe_workers.dispatch():
                if not done:
                    print("HAUKUR_DEBUG:", inputs, targets)
                    done = True
                yield {
                    "inputs": inputs,
                    "targets": targets,
                }
        else:
            for sample in self.generate_samples(data_dir, tmp_dir, dataset_split):
                yield {
                    "inputs": enc.encode(sample["inputs"]) + eos_list,
                    "targets": enc.encode(sample["targets"]) + eos_list,
                }

    def generate_samples(self, data_dir, tmp_dir, dataset_split,
            custom_iterator=text_problems.text2text_txt_iterator):
        def map_repeat_k(stream, k):
            for item in stream:
                yield from itertools.repeat(item, k)

        if dataset_split == problem.DatasetSplit.TRAIN:
            train_iter = custom_iterator("/data/en-is/v4/train/en-is.train.en", "/data/en-is/v4/train/en-is.train.is")
            # k=1 results in ~200mb dataset size on disk
            return map_repeat_k(train_iter, 50)  #  10 gb baked size
        return custom_iterator("/data/en-is/v4/dev/en-is.dev.bleu_sample.en", "/data/en-is/v4/dev/en-is.dev.bleu_sample.is")


@registry.register_problem
class TranslateEnis16kV4BpeBakedTmp(TranslateEnis16kV4BpeBaked):

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        vocabs = self.feature_encoders(data_dir)
        enc = vocabs["inputs"]
        eos_list = [EOS]
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        if dataset_split == problem.DatasetSplit.TRAIN:
            generator = itertools.islice(generator, 1000)
        for sample in generator:
            yield {
                "inputs": enc.encode(sample["inputs"]) + eos_list,
                "targets": enc.encode(sample["targets"]) + eos_list,
            }


@registry.register_problem
class TranslateEnis16kBt(ScorableTranslateProblem):

    @property
    def batch_size_means_tokens(self):
        return True

    @property
    def vocab_filename(self):
        return "vocab.translate_enis16k.16384.subwords"

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def serving_input_scoring_fn(self, data_dir, tmp_dir, force_get=False):
        """Input fn for serving export, when using TransformerScorer, starting from serialized example."""
        mode = tf.estimator.ModeKeys.PREDICT
        serialized_example = tf.placeholder(
            dtype=tf.string, shape=[None], name="serialized_example")
        dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
        dataset = dataset.map(self.decode_example)
        dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
        dataset = dataset.map(data_reader.cast_ints_to_int32)

        if use_tpu:
          padded_shapes = data_reader.pad_for_tpu(dataset.output_shapes, hparams,
                                                  hparams.max_length)
          batch_size = 1 if not decode_hparams else getattr(decode_hparams,
                                                            "batch_size", 1)
          dataset = dataset.padded_batch(
              batch_size, padded_shapes, drop_remainder=False)
          dataset = dataset.map(
              functools.partial(data_reader.pad_batch, batch_multiple=batch_size))
        else:
          dataset = dataset.padded_batch(
              tf.shape(serialized_example, out_type=tf.int64)[0],
              dataset.output_shapes)

        dataset = dataset.map(data_reader.standardize_shapes)
        features = tf.data.experimental.get_single_element(dataset)

        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=serialized_example)

    def source_data_files(self, dataset_split):
        data_files = [["rmh.bt.sample_3m.tar.gz", ("rmh.bt.sample_3m.eng_mt", "rmh.bt.sample_3m.isl")]]
        data_files.extend(super(TranslateEnis16kBt, self).source_data_files(dataset_split))
        return data_files


@registry.register_problem
class Rmh2017BtEnis16kUntagged(PartialTargetsTranslateProblem):
    """ translated with beam_size 4 """

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
        if dataset_split == problem.DatasetSplit.TRAIN:
            iter_ = custom_iterator("/data/datasets/rmh2017-bt/rmh2017_bt_40m.filtered.deduped.tsv")
            return iter_


@registry.register_problem
class Rmh2017BtEnis16kTagged(Rmh2017BtEnis16kUntagged):
    def __init__(self, was_reversed=False, was_copy=False):
        super(Rmh2017BtEnis16kTagged, self).__init__(was_reversed, was_copy)

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
        stream = super(Rmh2017BtEnis16kTagged, self).dataset(
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

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)


@registry.register_problem
class Rmh2017BtEnis16kSampledTemp1p0Tagged(PartialTargetsTranslateProblem):
    """ Translated via sampling (token level), sampling_temp=1.0 """

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
        if dataset_split == problem.DatasetSplit.TRAIN:
            iter_ = custom_iterator("/data/datasets/rmh2017-bt/rmh2017_bt_40m.sampled_1p0.tsv")
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
        stream = super(Rmh2017BtEnis16kSampledTemp1p0Tagged, self).dataset(
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
class TranslateEnis16kV4UntaggedBt(PartialTargetsTranslateProblem):

    @property
    def only_eval_first_problem(self):
        """Only run validation on examples from the first problem."""
        return False

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
        rmh = Rmh2017BtEnis16kUntagged()
        rmh.generate_data(data_dir, tmp_dir, task_id=task_id)

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
        enis = TranslateEnis16kV4().dataset(
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
        rmh = Rmh2017BtEnis16kUntagged().dataset(
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
        for dataset in [enis, rmh]:
            if is_training:
                dataset = task_dataset.repeat()

            datasets.append(task_dataset)

        sampled_dataset = tf.data.experimental.sample_from_datasets(
            datasets,
            weights=np.asarray([1, 1], dtype=np.float64))
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
        if mode == tf.estimator.ModeKeys.PREDICT:
            return super(TranslateEnis16kV4UntaggedBt, self).input_fn(
                mode,
                hparams,
                data_dir=data_dir,
                params=params,
                config=config,
                force_repeat=force_repeat,
                prevent_repeat=prevent_repeat,
                dataset_kwargs=dataset_kwargs,
                )
        if dataset_kwargs is not None and "data_dir" in dataset_kwargs and data_dir:
            dataset_kwargs.pop("data_dir")
        enis = TranslateEnis16kV4()
        enis = enis.input_fn(
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
            return enis

        rmh = Rmh2017BtEnis16kUntagged().input_fn(
                mode,
                hparams,
                data_dir=data_dir,
                params=params,
                config=config,
                force_repeat=force_repeat,
                prevent_repeat=prevent_repeat,
                dataset_kwargs=dataset_kwargs,
                )
        streams = []
        for stream in [enis, rmh]:
            if mode == tf.estimator.ModeKeys.PREDICT:
                pass

            streams.append(stream)

        sampled_dataset = tf.data.experimental.sample_from_datasets(
            streams,
            weights=np.asarray([1, 1], dtype=np.float64))
        return sampled_dataset

    def source_data_files(self, dataset_split):
        return []


@registry.register_problem
class TranslateEnis16kV4TaggedBt(TranslateEnis16kV4UntaggedBt):

    mixing_weigts = [1,1]
    other_problem = Rmh2017BtEnis16kTagged

    def __init__(self, was_reversed=False, was_copy=False):
        super(TranslateEnis16kV4TaggedBt, self).__init__(was_reversed, was_copy)

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
        vocab_size = prob_hp.vocabulary["inputs"].vocab_size
        if self.feature_encoders(data_dir)["inputs"].vocab_size == vocab_size:
            self._hparams.vocab_size["inputs"] = vocab_size + 1
            self._hparams.vocab_size["targets"] = vocab_size + 1
        task_id = self._hparams.vocab_size["inputs"] - 1
        enis = TranslateEnis16kV4()
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

        rmh = self.other_problem()
        rmh._hparams = self._hparams
        rmh._task_id = task_id
        rmh_stream = rmh.input_fn(
                mode,
                hparams,
                data_dir=data_dir,
                params=params,
                config=config,
                force_repeat=force_repeat,
                prevent_repeat=prevent_repeat,
                dataset_kwargs=dataset_kwargs,
                )
        streams = [enis_stream, rmh_stream]

        sampled_dataset = tf.data.experimental.sample_from_datasets(
            streams,
            weights=np.asarray(self.mixing_weights, dtype=np.float64))
        return sampled_dataset

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        enis = TranslateEnis16kV4()
        enis.generate_data(data_dir, tmp_dir, task_id=task_id)
        rmh = self.other_problem()
        rmh.generate_data(data_dir, tmp_dir, task_id=task_id)

    def source_data_files(self, dataset_split):
        return []

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return TranslateEnis16k().get_or_create_vocab(data_dir, tmp_dir, force_get=force_get)

    def get_hparams(self, model_hparams=None):
        prob_hp = super(TranslateEnis16kV4TaggedBt, self).get_hparams(model_hparams=model_hparams)
        vocab_size = prob_hp.vocabulary["inputs"].vocab_size
        vocab_size_with_task_id = 16421 + 1
        if vocab_size != vocab_size_with_task_id:
            self._hparams.vocab_size["inputs"] = vocab_size_with_task_id
            self._hparams.vocab_size["targets"] = vocab_size_with_task_id
        return prob_hp


@registry.register_problem
class TranslateEnis16kV4TaggedBtSampled1Mix1(TranslateEnis16kV4TaggedBt):
    mixing_weights = [1,1]
    other_problem = Rmh2017BtEnis16kSampledTemp1p0Tagged


@registry.register_problem
class TranslateEnis16kV4TaggedBtSampled1Mix2(TranslateEnis16kV4TaggedBt):
    mixing_weights = [15,85]
    other_problem = Rmh2017BtEnis16kSampledTemp1p0Tagged


@registry.register_hparams
def lstm_luong_attention_multi_shared_emb():
  """Multi-head Luong attention with shared input/target embeddings and weights."""
  hparams = lstm.lstm_luong_attention_multi()
  hparams.shared_embedding_and_softmax_weights = True
  hparams.hidden_size = 512
  hparams.attention_layer_size = 512
  return hparams


@registry.register_hparams
def lstm_luong_attention_multi_brnn():
  """Multi-head Luong attention with shared input/target embeddings and weights."""
  hparams = lstm.lstm_luong_attention_multi()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.hidden_size = 512
  hparams.attention_layer_size = hparams.hidden_size
  hparams.num_hidden_layers = 2
  hparams.batch_size = 600
  hparams.daisy_chain_variables = False
  hparams.no_data_parallelism = True
  hparams.max_length = 80  # override after warm-up
  hparams.eval_drop_long_sequences = True
  return hparams
