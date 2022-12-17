from pandas import DataFrame

from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
import re

import torch
from torch.utils.data import TensorDataset, random_split, Dataset, Sampler, DataLoader, Subset
from hydra_zen.typing import Partial
import pytorch_lightning as pl
# from hashformers import TransformerWordSegmenter as WordSegmenter


class EmotionDataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: str, max_input_length: int, remove_hashtags: bool, replace_mentions: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_input_length = max_input_length
        self.replace_mentions = replace_mentions
        self.emo2id = {emotion: i for i, emotion in enumerate(df["label"].unique())}
        self.remove_hashtags = remove_hashtags
        self.dataset = self.encode(df)

    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def encode(self, df: DataFrame):
        # list [ [input id, attention mask, label]]
        # TODO: preprocess the inputs
        # process the outputs


        # ws = WordSegmenter()
        regex_twitter_mentions = re.compile(r"\@.+?\b", flags=re.MULTILINE)
        if self.remove_hashtags:
            regex_hashtags = re.compile(r"\#.+?\b ", flags=re.MULTILINE)

        input_ids = []
        attention_masks = []
        labels = []

        for _, row in df.iterrows():
            label, sentence = row['label'], row['text']
            # process the input
            # print("debug-------------", sentence, type(sentence))
            if not isinstance(sentence, str):
                continue
            
            sentence = re.sub(
                pattern=regex_twitter_mentions,
                repl=self.replace_mentions,
                string=sentence
            )

            if self.remove_hashtags:
                sentence = re.sub(
                    pattern=regex_hashtags,  # type: ignore
                    repl="",
                    string=sentence
                )
            else:
                if "#" in sentence:
                    sentence = sentence.replace("#", "# ")
            

            encode_dict = self.tokenizer.encode_plus(
                text=sentence,
                add_special_tokens=True,
                max_length=self.max_input_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_id, att_mask = encode_dict['input_ids'], encode_dict['attention_mask']
            # process the output
            label_id = self.emo2id[label]

            input_ids.append(input_id)
            attention_masks.append(att_mask)
            labels.append(label_id)
        
        input_ids_tensor = torch.cat(input_ids, dim=0)
        attention_masks_tensor = torch.cat(attention_masks, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(input_ids_tensor.shape, attention_masks_tensor.shape, labels_tensor.shape)

        return TensorDataset(input_ids_tensor, attention_masks_tensor, labels_tensor)


class EmotionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: DataFrame,
        remove_hashtags: bool,
        replace_mentions: str,
        hf_checkpoint: str,
        batch_size: int,
        train_size: float,
        val_size: float,
        max_input_length: int,
        sampler: Partial[Sampler],
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.df = df
        self.hf_checkpoint = hf_checkpoint
        self.batch_size = batch_size
        self.train_size = train_size
        self.remove_hashtags = remove_hashtags
        self.val_size = val_size
        self.max_input_length = max_input_length
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.replace_mentions = replace_mentions

        self.tokenizer: PreTrainedTokenizer = None  # type: ignore
        self.train: Subset[EmotionDataset] = None  # type: ignore
        self.val: Subset[EmotionDataset] = None  # type: ignore
        self.test: Subset[EmotionDataset] = None  # type: ignore
     
    @property
    def n_classes(self) -> int:
        return len(self.df["label"].unique())
     
    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(self.hf_checkpoint)
    
    def setup(self, stage: str) -> None:
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_checkpoint)  # type: ignore
        
        dataset = EmotionDataset(
            df=self.df,
            tokenizer=self.hf_checkpoint,
            max_input_length=self.max_input_length,
            remove_hashtags=self.remove_hashtags,
            replace_mentions=self.replace_mentions
        )

        self.train, self.val, self.test = random_split(
            dataset=dataset,
            lengths=[
                self.train_size,
                self.val_size,
                1 - self.train_size - self.val_size
            ]
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            sampler=self.sampler(self.train),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            sampler=self.sampler(self.val),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            sampler=self.sampler(self.test),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
