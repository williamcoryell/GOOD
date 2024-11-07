import autocuda
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
)
from pathlib import Path
from typing import Union
import os
from findfile import find_file
import google.protobuf.internal


class CheckpointManager:
    def parse_checkpoint(
        self,
        checkpoint: Union[str, Path] = None,
        task_code: str = "ACOS",
    ) -> Union[str, Path]:
        """
        Parse a given checkpoint file path or name and returns the path of the checkpoint directory.

        Args:
            checkpoint (Union[str, Path], optional): Zipped checkpoint name, checkpoint path, or checkpoint name queried from Google Drive. Defaults to None.
            task_code (str, optional): Task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc. Defaults to TaskCodeOption.Aspect_Polarity_Classification.

        Returns:
            Path: The path of the checkpoint directory.

        Example:
            ```
            manager = CheckpointManager()
            checkpoint_path = manager.parse_checkpoint("checkpoint.zip", "apc")
            ```
        """
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            # directly load checkpoint from local path
            if os.path.exists(checkpoint):
                return checkpoint

            if find_file(os.getcwd(), [checkpoint, task_code, ".config"]):
                # load checkpoint from current working directory with task specified
                checkpoint_config = find_file(
                    os.getcwd(), [checkpoint, task_code, ".config"]
                )
            else:
                # load checkpoint from current working directory without task specified
                checkpoint_config = find_file(os.getcwd(), [checkpoint, ".config"])

            if checkpoint_config:
                # locate the checkpoint directory
                checkpoint = os.path.dirname(checkpoint_config)

        return checkpoint


class T5Generator:
    def __init__(self, checkpoint):
        try:
            checkpoint = CheckpointManager().parse_checkpoint(checkpoint, "ACOS")
        except Exception as e:
            print(e)

        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model.config.max_length = 128
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = autocuda.auto_cuda()
        self.model.to(self.device)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample["text"], max_length=1024, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        # Set training arguments
        args = Seq2SeqTrainingArguments(**kwargs)

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
            if tokenized_datasets.get("test") is not None
            else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print("\nModel training started ....")
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def predict(self, text, **kwargs):
        """
        Predict the output from the model.
        """
        ate_instructor = ATEInstruction()
        apc_instructor = APCInstruction()
        op_instructor = OpinionInstruction()
        cat_instructor = CategoryInstruction()
        result = {
            "text": text,
        }

        # ATE inference
        inputs = self.tokenizer(
            ate_instructor.prepare_input(text), truncation=True, return_tensors="pt"
        ).to(self.device)
        ate_outputs = self.model.generate(**inputs, **kwargs)
        ate_outputs = self.tokenizer.batch_decode(
            ate_outputs, skip_special_tokens=True
        )[0]
        result["aspect"] = [asp.strip() for asp in ate_outputs.split("|")]

        # APC inference
        inputs = self.tokenizer(
            apc_instructor.prepare_input(text, ate_outputs),
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        apc_outputs = self.model.generate(**inputs, **kwargs)
        apc_outputs = self.tokenizer.batch_decode(
            apc_outputs, skip_special_tokens=True
        )[0]
        result["polarity"] = [sent.strip() for sent in apc_outputs.split("|")]

        # Opinion inference
        inputs = self.tokenizer(
            op_instructor.prepare_input(text, ate_outputs),
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        op_outputs = self.model.generate(**inputs, **kwargs)
        op_outputs = self.tokenizer.batch_decode(op_outputs, skip_special_tokens=True)[
            0
        ]
        result["opinion"] = [op.strip() for op in op_outputs.split("|")]

        # Category inference
        inputs = self.tokenizer(
            cat_instructor.prepare_input(text, ate_outputs),
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        cat_outputs = self.model.generate(**inputs, **kwargs)
        cat_outputs = self.tokenizer.batch_decode(
            cat_outputs, skip_special_tokens=True
        )[0]
        result["category"] = [cat.strip() for cat in cat_outputs.split("|")]
        ensemble_result = {
            "text": text,
            "Quadruples": [
                {
                    "aspect": asp,
                    "polarity": sent.partition(":")[2],
                    "opinion": op.partition(":")[2],
                    "category": cat.partition(":")[2],
                }
                for asp, sent, op, cat in zip(
                    result["aspect"],
                    result["polarity"],
                    result["opinion"],
                    result["category"],
                )
            ],
        }
        print(ensemble_result)
        return ensemble_result

    def get_labels(
        self,
        tokenized_dataset,
        trained_model_path=None,
        predictor=None,
        batch_size=4,
        sample_set="train",
    ):
        """
        Get the predictions from the trained model.
        """
        if not predictor:
            print("Prediction from checkpoint")

            def collate_fn(batch):
                input_ids = [torch.tensor(example["input_ids"]) for example in batch]
                input_ids = pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                return input_ids

            dataloader = DataLoader(
                tokenized_dataset[sample_set],
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
            predicted_output = []
            self.model.to(self.device)
            print("Model loaded to: ", self.device)

            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output_ids = self.model.generate(batch)
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for output_text in output_texts:
                    predicted_output.append(output_text)
        else:
            print("Prediction from trainer")
            output_ids = predictor.predict(
                test_dataset=tokenized_dataset[sample_set]
            ).predictions
            predicted_output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
        return predicted_output

    def get_aspect_metrics(self, true_aspects, pred_aspects):
        aspect_p = precision_score(true_aspects, pred_aspects, average="macro")
        aspect_r = recall_score(true_aspects, pred_aspects, average="macro")
        aspect_f1 = f1_score(true_aspects, pred_aspects, average="macro")
        return aspect_p, aspect_r, aspect_f1

    def get_classic_metrics(self, y_true, y_pred):
        for i in range(len(y_true)):
            y_true[i] = y_true[i].replace(" ", "")
            y_pred[i] = y_pred[i].replace(" ", "")
            print(y_true[i])
            print(y_pred[i])
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, force_download=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_checkpoint, force_download=True
        )
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample["input_ids"] = self.tokenizer(
            sample["text"], max_length=1024, truncation=True
        ).input_ids
        sample["labels"] = self.tokenizer(
            sample["labels"], max_length=128, truncation=True
        ).input_ids
        return sample

    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(**kwargs)

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
            if tokenized_datasets.get("test") is not None
            else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print("\nModel training started ....")
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(
        self, tokenized_dataset, predictor=None, batch_size=4, sample_set="train"
    ):
        """
        Get the predictions from the trained model.
        """
        if not predictor:
            print("Prediction from checkpoint")

            def collate_fn(batch):
                input_ids = [torch.tensor(example["input_ids"]) for example in batch]
                input_ids = pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                return input_ids

            dataloader = DataLoader(
                tokenized_dataset[sample_set],
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
            predicted_output = []
            self.model.to(self.device)
            print("Model loaded to: ", self.device)

            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output_ids = self.model.to.generate(batch)
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for output_text in output_texts:
                    predicted_output.append(output_text)
        else:
            print("Prediction from trainer")
            pred_proba = predictor.predict(
                test_dataset=tokenized_dataset[sample_set]
            ).predictions[0]
            output_ids = np.argmax(pred_proba, axis=2)
            predicted_output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
        return predicted_output

    def get_metrics(self, y_true, y_pred):
        cnt = 0
        for gt, pred in y_true, y_pred:
            if gt == pred:
                cnt += 1
        return cnt / len(y_true)


class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class ATEInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
{self.eos_instruction}
aspect:battery life|aspect:cord

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
{self.eos_instruction}
aspect:food|aspect:menu|aspect:service|aspect:setting

Now extract aspects from the following example:
input: """

        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract aspects one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text):
        return self.bos_instruction + input_text + self.eos_instruction


class APCInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding polarity. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:positive|cord:positive

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food, menu, service, setting
{self.eos_instruction}
food:positive|menu:positive|service:positive|setting:positive
    
Now predict aspect sentiments from the following example:

input: """
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us predict sentiments one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )


class OpinionInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding polarity. Here are some examples:

example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:good|cord:NULL
    
example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food, menu, service, setting
{self.eos_instruction}
food:great|menu:good|service:great|setting:unpretensious

Now extract opinions for the following example:
input:"""
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract opinions one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )


class CategoryInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product or service. The task is to extract the aspects and their corresponding categories. Here are some examples:
    
example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
The aspects are: battery life, cord
{self.eos_instruction}
battery life:POWER_SUPPLY#GENERAL|cord:NULL

example 2-
input: Great food, good size menu, great service and an unpretensious setting.
The aspects are: food:FOOD#QUALITY| menu:RESTAURANT#GENERAL|service:SERVICE#GENERAL|setting:SERVICE#GENERAL
{self.eos_instruction}
food:FOOD#QUALITY, menu:RESTAURANT#GENERAL, service:SERVICE#GENERAL, setting:SERVICE#GENERAL

Now extract categories for the following example:
input: """
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract chategories one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, aspects):
        return (
            self.bos_instruction
            + input_text
            + f"The aspects are: {aspects}"
            + self.eos_instruction
        )
    

T5Classifier("checkpoint.config")