#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import os

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Callable, Dict, Tuple, Any, Optional
import numpy as np
from torch.cuda.amp import autocast

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from transformers.modeling_utils import unwrap_model
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_pt_utils import IterableDatasetShard

from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer

from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder
from extraction.extraction_metrics import get_extract_metrics
from seq2seq.label_smoother_sum import SumLabelSmoother
from seq2seq.utils import lmap

from prefix.prefix_model import PrefixEncoderDecoder

logger = logging.getLogger(__name__)


def add_logging_file(training_args):
    fh = logging.FileHandler(os.path.join(training_args.output_dir.rstrip(os.sep) + '.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def decode_tree_str(sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
                    tokenizer: PreTrainedTokenizer) -> List[str]:
    def clean_tree_text(x):
        return x.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
    
    sequences = np.where(sequences != -100, sequences, tokenizer.pad_token_id)
    
    str_list = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    return lmap(clean_tree_text, str_list)


def build_compute_extract_metrics_event_fn(decoding_type_schema: EventSchema,
                                           decoding_format: str,
                                           tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)
    
    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        return decode_tree_str(pred.predictions, tokenizer), decode_tree_str(pred.label_ids, tokenizer)
    
    def extraction_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        extraction = get_extract_metrics(pred_lns=pred_str, tgt_lns=label_str, label_constraint=decoding_type_schema,
                                         decoding_format=decoding_format)
        # rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        extraction.update({"gen_len": summ_len})
        # extraction.update( )
        return extraction
    
    compute_metrics_fn = extraction_metrics
    return compute_metrics_fn


@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    label_smoothing_sum: bool = field(default=False,
                                      metadata={"help": "Whether to use sum token loss for label smoothing"})
    tuning_type: str = field(default="both", metadata={"help": "tuning type in both, prefix or fine-tuning"})
    prefix_len: int = field(default=5, metadata={"help": "length of prefix tokens"})
    is_knowledge: bool = field(default=False, metadata={"help": "use knowledge-enhanced prompt generation or not."})
    no_module: bool = field(default=False, metadata={"help": "use nn to generate prefix or not"})
    predict_with_generate: bool = field(default=False, metadata={"help": "use predict with generate"})
    

class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, decoding_format='tree', source_prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema
        
        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_sum and self.args.label_smoothing_factor != 0:
            self.label_smoother = SumLabelSmoother(epsilon=self.args.label_smoothing_factor)
            print('Using %s' % self.label_smoother)
        elif self.args.label_smoothing_factor != 0:
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None
        
        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix)
        else:
            self.constraint_decoder = None
    
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        
        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)
        
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if self.constraint_decoder else None,
            )
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }
        
        generated_tokens = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None
        
        if self.args.prediction_loss_only:
            return loss, None, None
        
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        
        return loss, generated_tokens, labels
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        print("constrained trainer 222:", len(train_dataset))
        # if super.is_datasets_available() and isinstance(train_dataset, super.datasets.Dataset):
        #     print("constrained trainer 223:", True)
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            print("constrained trainer 227:", True)
            if self.args.world_size > 1:
                print("constrained trainer 229:", True)
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        train_sampler = self._get_train_sampler()
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        # if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
        #     test_dataset = self._remove_unused_columns(test_dataset, description="test")
        
        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        test_sampler = self._get_eval_sampler(test_dataset)
        
        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        eval_sampler = self._get_eval_sampler(eval_dataset)
        
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
