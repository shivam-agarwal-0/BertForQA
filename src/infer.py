import time
start = time.time()

import config
import utils
from model import SQuAD2Model

import numpy as np
import torch
from transformers import BertConfig

model_config = BertConfig.from_pretrained('bert-base-uncased')
model_config.output_hidden_states = True
model = SQuAD2Model(conf=model_config)

device = config.DEVICE
checkpoint = torch.load(f'{config.MODEL_PATH}\\model_2.pth', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

input = utils.convert_input_to_features(config.CONTEXT, config.QUESTION)
input_ids, attention_masks, token_type_ids = input
input_ids = input_ids.unsqueeze(0).to(device)
attention_masks = attention_masks.unsqueeze(0).to(device)
token_type_ids = token_type_ids.unsqueeze(0).to(device)

outputs_start, outputs_end = model(input_ids, attention_masks, token_type_ids)

outputs_start = outputs_start.squeeze().detach().cpu().numpy()
outputs_end = outputs_end.squeeze().detach().cpu().numpy()

outputs_start = np.argmax(outputs_start[1:])
outputs_end = np.argmax(outputs_end[1:])

prediction = utils.convert_predictions_to_text(input_ids[0], outputs_start, outputs_end)

print(prediction)
print(f'TOTAL TIME TAKEN: {time.time() - start} seconds')
