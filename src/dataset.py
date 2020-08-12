import config

import torch
from torch.utils.data import Dataset

class SQuAD2Dataset(Dataset):
	def __init__(self, df):
		self.df = df
		self.tokenizer = config.TOKENIZER
		self.max_len = config.MAX_LEN

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		context = self.df['context'][idx]
		question = self.df['question'][idx]
		answer = self.df['answers'][idx]['text']
		answer_start = self.df['answers'][idx]['answer_start']
		is_impossible = self.df['is_impossible'][idx]
		qas_id = self.df['id'][idx]

		input_text = '[CLS] ' + question + ' [SEP] ' + context + ' [SEP]'
		input_tokens = self.tokenizer.tokenize(input_text)

		start_position = 0
		end_position = 0

		if not is_impossible:
			answer_tokens = self.tokenizer.tokenize(answer)

			for i, tok in enumerate(input_tokens):
				if tok == answer_tokens[0] and input_tokens[min(len(input_tokens)-1, i+len(answer_tokens)-1)] == answer_tokens[-1]:
					start_position = i
					end_position = min(len(input_tokens)-1, i+len(answer_tokens)-1)
					if start_position >= max_len or end_position >= max_len:
						start_position = 0
						end_position = 0

		token_type_ids = [0] * (len(self.tokenizer.tokenize(question)) + 2)
		token_type_ids = token_type_ids + [1] * (len(input_tokens) - len(token_type_ids))

		input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
		attention_mask = [1] * len(input_tokens)

		# Truncate
		if len(input_ids) > self.max_len:
			input_ids = input_ids[:self.max_len]
			attention_mask = attention_mask[:self.max_len]
			token_type_ids = token_type_ids[:self.max_len]

		# Padding
		if len(input_ids) < self.max_len:
			input_ids = input_ids + [0] * (max_len - len(input_tokens))
			attention_mask = attention_mask + [0] * (max_len - len(input_tokens))
			token_type_ids = token_type_ids + [0] * (max_len - len(input_tokens))

		# convert lists to tensor
		input_ids = torch.tensor(input_ids, dtype=torch.long)
		attention_mask = torch.tensor(attention_mask, dtype=torch.float)
		token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
		start_position = torch.tensor(start_position, dtype=torch.long).unsqueeze(0)
		end_position = torch.tensor(end_position, dtype=torch.long).unsqueeze(0)

		return input_ids, attention_mask, token_type_ids, start_position, end_position
