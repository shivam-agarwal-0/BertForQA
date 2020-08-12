import config
import utils
from dataset import SQuAD2Dataset
from model import SQuAD2Model

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig

train_df = utils.load_json_as_pandas_df(train=True)
valid_df = utils.load_json_as_pandas_df(train=False)

device = config.DEVICE

train_dataset = SQuAD2Dataset(train_df)
valid_dataset = SQuAD2Dataset(valid_df)

train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

model_config = BertConfig.from_pretrained('bert-base-uncased')
model_config.output_hidden_states = True
model = SQuAD2Model(conf=model_config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(config.EPOCHS):

    model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        input_ids, attention_masks, token_type_ids, start_positions, end_positions = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        start_positions = start_positions.squeeze().to(device)
        end_positions = end_positions.squeeze().to(device)

        outputs_start, outputs_end = model(input_ids, attention_masks, token_type_ids)

        loss_start = criterion(outputs_start, start_positions)
        loss_end = criterion(outputs_end, end_positions)
        total_loss = loss_start + loss_end
        total_loss.backward()

        optimizer.step()
        if (step+1) % 1000 == 0:
            print(f'Epoch: {epoch+1} || Step: {step+1} || Training Loss: {total_loss.item()}')

    model.eval()

    true_start = []
    true_end = []
    pred_start = []
    pred_end = []

    with torch.no_grad():

        for batch in tqdm(valid_dataloader):

            input_ids, attention_masks, token_type_ids, start_positions, end_positions = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            start_positions = start_positions.squeeze().to(device)
            end_positions = end_positions.squeeze().to(device)

            outputs_start, outputs_end = model(input_ids, attention_masks, token_type_ids)

            true_start.append(start_positions)
            true_end.append(end_positions)
            pred_start.append(outputs_start)
            pred_end.append(outputs_end)

    true_start = torch.cat(true_start)
    true_end = torch.cat(true_end)
    pred_start = torch.cat(pred_start)
    pred_end = torch.cat(pred_end)

    loss = criterion(pred_start, true_start) + criterion(pred_end, true_end)

    print(f'Validation Loss: {loss.item()}')
    
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()},
               f'{config.MODEL_PATH}/model_{epoch+1}.pth')
