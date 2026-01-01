import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DebertaV2Model,
    DebertaV2Config,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from .modules import BiAttention, get_aggregated


class DebertaConfigForWebshop(PretrainedConfig):
    model_type = "deberta-v2"

    def __init__(
        self,
        pretrained=True,
        image=False,
        **kwargs
    ):
        self.pretrained = pretrained
        self.image = image
        super().__init__(**kwargs)


class DebertaModelForWebshop(PreTrainedModel):
    config_class = DebertaConfigForWebshop

    def __init__(self, config):
        super().__init__(config)
        if config.pretrained:
            self.encoder = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        else:
            deberta_config = DebertaV2Config.from_pretrained('microsoft/deberta-v3-large')
            self.encoder = DebertaV2Model(deberta_config)
        
        hidden_size = 1024  # deberta-v3-large hidden size
        
        self.attn = BiAttention(hidden_size, 0.0)
        self.linear_1 = nn.Linear(hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, 1)
        
        if config.image:
            self.image_linear = nn.Linear(512, hidden_size)
        else:
            self.image_linear = None

        self.linear_3 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        sizes = sizes.tolist()
        
        state_rep = self.encoder(state_input_ids, attention_mask=state_attention_mask)[0]
        
        if images is not None and self.image_linear is not None:
            images = self.image_linear(images)
            state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
            state_attention_mask = torch.cat([state_attention_mask[:, :1], state_attention_mask], dim=1)
        
        action_rep = self.encoder(action_input_ids, attention_mask=action_attention_mask)[0]
        
        state_rep = torch.cat([state_rep[i:i+1].repeat(j, 1, 1) for i, j in enumerate(sizes)], dim=0)
        state_attention_mask = torch.cat([state_attention_mask[i:i+1].repeat(j, 1) for i, j in enumerate(sizes)], dim=0)
        
        act_lens = action_attention_mask.sum(1).tolist()
        state_action_rep = self.attn(action_rep, state_rep, state_attention_mask)
        state_action_rep = self.relu(self.linear_1(state_action_rep))
        act_values = get_aggregated(state_action_rep, act_lens, 'mean')
        act_values = self.linear_2(act_values).squeeze(1)

        logits = [F.log_softmax(_, dim=0) for _ in act_values.split(sizes)]

        loss = None
        if labels is not None:
            loss = - sum([logit[label] for logit, label in zip(logits, labels)]) / len(logits)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def rl_forward(self, state_batch, act_batch, value=False, q=False, act=False):
        act_values = []
        act_sizes = []
        values = []
        for state, valid_acts in zip(state_batch, act_batch):
            with torch.set_grad_enabled(not act):
                state_ids = torch.tensor([state.obs]).cuda()
                state_mask = (state_ids > 0).int()
                act_ids = [torch.tensor(_) for _ in valid_acts]
                act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True).cuda()
                act_mask = (act_ids > 0).int()
                act_size = torch.tensor([len(valid_acts)]).cuda()
                if self.image_linear is not None:
                    images = [state.image_feat]
                    images = [torch.zeros(512) if _ is None else _ for _ in images]
                    images = torch.stack(images).cuda()
                else:
                    images = None
                logits = self.forward(state_ids, state_mask, act_ids, act_mask, act_size, images=images).logits[0]
                act_values.append(logits)
                act_sizes.append(len(valid_acts))
            if value:
                v = self.encoder(state_ids, state_mask)[0]
                values.append(self.linear_3(v[0][0]))
        act_values = torch.cat(act_values, dim=0)
        act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)
        if value:
            values = torch.cat(values, dim=0)
            return act_values, act_sizes, values
        else:
            return act_values, act_sizes
