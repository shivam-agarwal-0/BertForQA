from torch import nn
from transformers import BertPreTrainedModel, BertModel

class SQuAD2Model(BertPreTrainedModel):
	def __init__(self, conf):
		super(SQuAD2Model, self).__init__(conf)
		self.bert = BertModel(conf)
		self.drop = nn.Dropout(0.1)
		self.fc = nn.Linear(768, 2)
	
	def forward(self, ids, mask, token_type_ids):
		_, _, out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

		out = self.drop(out[0])
		logits = self.fc(out)

		start_logits, end_logits = logits.split(1, dim=-1)

		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)

		return start_logits, end_logits
