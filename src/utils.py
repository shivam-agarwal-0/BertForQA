import config

import pandas as pd
import torch

def load_json_as_pandas_df(train=False):
	if train:
		df = pd.read_json(config.TRAINING_FILE, orient='records')
	else:
		df = pd.read_json(config.VALIDATION_FILE, orient='records')
	
	df = pd.DataFrame.from_records(df['data'])
	df = df.explode('paragraphs').reset_index(drop=True)
	df['context'] = df['paragraphs'].apply(lambda x: x['context'])
	df['qas'] = df['paragraphs'].apply(lambda x: x['qas'])
	df = df[['title', 'context', 'qas']]
	df = df.explode('qas').reset_index(drop=True)
	df = pd.concat([df, pd.DataFrame.from_records(df['qas'])], axis=1).drop(['qas'], axis=1)
	df.loc[df['is_impossible'], 'answers'] = df.loc[df['is_impossible'], 'plausible_answers']
	df = df.drop(['plausible_answers'], axis=1)
	df['answers'] = df['answers'].apply(lambda x: x[0] if len(x) else {'text': '', 'answer_start': 0})

	return df

def convert_input_to_features(context, question):
	tokenizer = config.TOKENIZER
	max_len = config.MAX_LEN

	input_text = '[CLS] ' + question + ' [SEP] ' + context + ' [SEP]'
	input_tokens = tokenizer.tokenize(input_text)
	
	token_type_ids = [0] * (len(tokenizer.tokenize(question)) + 2)
	token_type_ids = token_type_ids + [1] * (len(input_tokens) - len(token_type_ids))

	input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
	attention_mask = [1] * len(input_tokens)

	# Truncate
	if len(input_ids) > max_len:
		input_ids = input_ids[:max_len]
		attention_mask = attention_mask[:max_len]
		token_type_ids = token_type_ids[:max_len]
		
	# Padding
	if len(input_ids) < max_len:
		input_ids = input_ids + [0] * (max_len - len(input_tokens))
		attention_mask = attention_mask + [0] * (max_len - len(input_tokens))
		token_type_ids = token_type_ids + [0] * (max_len - len(input_tokens))
	
	# convert lists to tensor
	input_ids = torch.tensor(input_ids, dtype=torch.long)
	attention_mask = torch.tensor(attention_mask, dtype=torch.float)
	token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
	
	return input_ids, attention_mask, token_type_ids

def convert_predictions_to_text(input_ids, start, end):
	if start == 0 or end == 0 or end < start:
		return 'ANSWER NOT FOUND IN THE PARAGRAPH!'

	tokenizer = config.TOKENIZER
	output_ids = input_ids[start: end+1]

	return tokenizer.decode(output_ids)