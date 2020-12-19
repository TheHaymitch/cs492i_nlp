from torch import nn
from transformers import *
from torch.nn import CrossEntropyLoss

'''
본 스크립트는 다음의 파일을 바탕으로 작성 됨
https://github.com/cooelf/AwesomeMRC/blob/9ff6dfe3c183d7ae8c7ad46ae88e19bf2c352098/transformer-mrc/transformers/modeling_albert.py
'''

class ElectraForRetroReader(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForRetroReader, self).__init__(config)
        self.num_labels = config.num_labels
        # config.hidden_dropout_prob=0.5

        self.electra = ElectraModel(config)
        self.answer = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.answer(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            # intensive reading - learning the actual span of the answer (start, end span)
            total_loss = (start_loss + end_loss)/2
            # sketchy reading - learning only the answerability of the question
            # total_loss = choice_loss
            outputs = (total_loss,) + outputs
        return outputs 