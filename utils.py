from datasets import load_from_disk

import copy
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
}

class InstructDataset(Dataset):
    def __init__(self, json_list, tokenizer, test_set=False, ignore_index=None):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []

        for j in tqdm(json_list):
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、
            # inputカラムありなしでテンプレート文を分けている。
            if 'input' in j:
                source_text = PROMPT_DICT['prompt_input'].format_map(j)
            else:
                source_text = PROMPT_DICT['prompt_no_input'].format_map(j)

            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            if not test_set:
                example_text = source_text + j['output'] + self.tokenizer.eos_token
            else:
                example_text = source_text + self.tokenizer.eos_token


            # 指示文のみ（「以下は、タスクを〜### 応答:」まで）をtokenize
            # ほしいのは指示文のlength
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )

            # 指示文と回答文を全てtokenize
            example_tokenized = self.tokenizer(
                example_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = example_tokenized['input_ids'][0]

            # LLMが生成してほしい正解の文章として入力文をそのままコピーする
            labels = copy.deepcopy(input_ids)

            # 指示文までの長さ
            source_len = source_tokenized['length'][0]

            # LLMに生成してほしい正解文章に指示文も含まれているので、
            # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
            labels[:source_len] = self.ignore_index

            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class InstructTestDataset(Dataset):
    def __init__(self, json_list, tokenizer, ignore_index=None):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []

        for j in tqdm(json_list):
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、
            # inputカラムありなしでテンプレート文を分けている。
            if 'input' in j:
                source_text = PROMPT_DICT['prompt_input'].format_map(j)
            else:
                source_text = PROMPT_DICT['prompt_no_input'].format_map(j)

            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            example_text = source_text + j['output'] + self.tokenizer.eos_token

            # 指示文のみ（「以下は、タスクを〜### 応答:」まで）をtokenize
            # ほしいのは指示文のlength
            source_tokenized = self.tokenizer(
                source_text + self.tokenizer.eos_token,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )

            # 指示文と回答文を全てtokenize
            example_tokenized = self.tokenizer(
                example_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            #input_ids = example_tokenized['input_ids'][0]
            input_ids = source_tokenized['input_ids'][0] # No need "output"

            # LLMが生成してほしい正解の文章として入力文をそのままコピーする
            #labels = copy.deepcopy(input_ids)
            labels = copy.deepcopy(example_tokenized['input_ids'][0])

            # 指示文までの長さ
            source_len = source_tokenized['length'][0] - 1

            # LLMに生成してほしい正解文章に指示文も含まれているので、
            # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
            labels[:source_len] = self.ignore_index

            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class InstructCollator():
    def __init__(self, tokenizer, ignore_index=None):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


