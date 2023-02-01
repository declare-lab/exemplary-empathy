import json
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import T5EncoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ERGMainModel(nn.Module):

    def __init__(self, base_model_name, max_source_length, max_target_length, strategy, exemplars=False, max_exemplars=None, fixed=False):
        super().__init__()
        self.erg_model = ERGModel(base_model_name, max_source_length, max_target_length, exemplars, max_exemplars)
        
        self.empathy_classifier_model1 = T5EncoderClassifier("base", 2, strategy)
        self.empathy_classifier_model1.load_state_dict(torch.load("saved/empathy/1619600015/model.pt"))

        self.empathy_classifier_model2 = T5EncoderClassifier("base", 2, strategy)
        self.empathy_classifier_model2.load_state_dict(torch.load("saved/empathy/1619600805/model.pt"))

        self.empathy_classifier_model3 = T5EncoderClassifier("base", 2, strategy)
        self.empathy_classifier_model3.load_state_dict(torch.load("saved/empathy/1619601340/model.pt"))

        self.sentiment_regression_model = T5EncoderRegressor("base", strategy)
        self.sentiment_regression_model.load_state_dict(torch.load("saved/sentiment/1620236944/model.pt"))

        self.fixed = fixed
        if self.fixed:
            for param in self.empathy_classifier_model1.parameters():
                param.requires_grad = False
            for param in self.empathy_classifier_model2.parameters():
                param.requires_grad = False
            for param in self.empathy_classifier_model3.parameters():
                param.requires_grad = False
            for param in self.sentiment_regression_model.parameters():
                param.requires_grad = False

    def forward(self, context, response, exemplars, padding=True, ignore_pad_token_for_loss=True):
        output, tokenized_response = self.erg_model(context, response, exemplars, padding, ignore_pad_token_for_loss)
        logits = output["logits"]
        response_mask = tokenized_response["attention_mask"]
        merged_context = [" ".join(conv) for conv in context]

        if self.fixed:
            self.empathy_classifier_model1.eval()
            self.empathy_classifier_model2.eval()
            self.empathy_classifier_model3.eval()
            self.sentiment_regression_model.eval()

        empathy1_preds = self.empathy_classifier_model1.output_from_logits(merged_context, logits, response_mask)
        empathy2_preds = self.empathy_classifier_model2.output_from_logits(merged_context, logits, response_mask)
        empathy3_preds = self.empathy_classifier_model3.output_from_logits(merged_context, logits, response_mask)
        sentiment_preds = self.sentiment_regression_model.output_from_logits(logits, response_mask)
        return output, empathy1_preds, empathy2_preds, empathy3_preds, sentiment_preds

class ERGModel(nn.Module):

    def __init__(self, base_model_name, max_source_length, max_target_length,
            exemplars=False, max_exemplars=None):
        super(ERGModel, self).__init__()
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        assert "t5" in base_model_name
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        
        # self.base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)

        with open("t5-config/" + base_model_name + ".json", "r") as f:
            self.base_model = T5ForConditionalGeneration(T5Config(**json.load(f)))
        model_weights = self.base_model.state_dict()
        for key in model_weights:
            if len(model_weights[key].shape)>=2:
                model_weights[key] = 2*torch.rand(model_weights[key].size()) - 1

        self.speaker_embedding = nn.Embedding(3, self.base_model.encoder.embed_tokens.embedding_dim,
                    padding_idx=0)
        self.speaker_embedding.weight.requires_grad = True
        if exemplars:
            self.exemplar_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            self.exemplar_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            self.transform_to_t5_decoder = nn.Linear(self.base_model.encoder.embed_tokens.embedding_dim + self.exemplar_model.encoder.embed_tokens.embedding_dim, self.base_model.encoder.embed_tokens.embedding_dim)
            self.max_exemplars = 999 if max_exemplars is None else max_exemplars
            for param in self.exemplar_model.parameters():
                param.requires_grad = False

    def _get_speaker_mask(self, speaker_span_lens, mask_length):
        speaker_mask = []
        for diag in speaker_span_lens:
            diag_mask = []
            speaker = 0
            for utt_len in diag:
                diag_mask += [speaker + 1] * utt_len
                speaker = (speaker + 1) % 2
            if mask_length >= len(diag_mask):
                diag_mask += [0] * (mask_length - len(diag_mask))
            else:
                diag_mask = diag_mask[:mask_length]
            speaker_mask.append(diag_mask)
        return torch.tensor(speaker_mask).to(self.speaker_embedding.weight.device)

    def _tokenize_input(self, context, max_length, padding=True, truncation=True):
        # Append prefix
        # context = [prefix + inp for inp in context]
        cat_context = [" ".join(inp) for inp in context]
        speaker_span_lens = []
        for diag in context:
            tokenized = self.tokenizer(diag)
            speaker_span_lens.append([len(utt)-1 for utt in tokenized["input_ids"]])
        # Setup the tokenizer for source
        inputs = self.tokenizer(cat_context, max_length=self.max_source_length, padding=padding, truncation=truncation, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].to(self.speaker_embedding.weight.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.speaker_embedding.weight.device)
        inputs["speaker_mask"] = self._get_speaker_mask(speaker_span_lens, inputs["attention_mask"].shape[1])
        return inputs

    def _preprocess(self, context, response=None, padding=True, ignore_pad_token_for_loss=True):
        "Preprocess data"

        inputs = self._tokenize_input(context, max_length=self.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        # with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(text_target=response, max_length=self.max_target_length, padding=padding, truncation=True, return_tensors="pt")

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding and ignore_pad_token_for_loss:
            labels["input_ids"] = torch.tensor([
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]).to(self.speaker_embedding.weight.device)
            
        labels["attention_mask"] = labels["attention_mask"].to(self.speaker_embedding.weight.device)
        return inputs, labels

    def _exemplar_mean_pool_representation(self, exemplars):
        exemplar_representations = []
        for samples in exemplars:
            samples = [f"paraphrase: {x}" for x in samples[:self.max_exemplars]]
            exemplars_inputs = self.exemplar_tokenizer(samples, max_length=self.max_source_length, padding=True, truncation=True, return_tensors="pt")
            exemplar_input_ids, exemplar_attention_masks = exemplars_inputs["input_ids"].to(self.speaker_embedding.weight.device), exemplars_inputs["attention_mask"].to(self.speaker_embedding.weight.device)
            encoder_output = self.exemplar_model.encoder(
                input_ids=exemplar_input_ids,
                attention_mask=exemplar_attention_masks
                )
            hidden_states = encoder_output[0] * exemplar_attention_masks[:, :, None]
            hidden_states = torch.sum(hidden_states, dim=1) / torch.sum(exemplar_attention_masks, dim=1, keepdim=True)
            hidden_states = torch.mean(hidden_states, dim=0)
            exemplar_representations.append(hidden_states)
        return torch.stack(exemplar_representations, 0)

    def forward(self, context, response, exemplars=None, padding=True, ignore_pad_token_for_loss=True):
        # assert exemplars is None or (hasattr(self, "exemplar_tokenizer") and hasattr(self, "exemplar_model"))
        inputs, labels = self._preprocess(context, response, padding, ignore_pad_token_for_loss)
        self.exemplar_model.eval()
        inputs_embeds = self.base_model.encoder.embed_tokens(inputs["input_ids"]) + self.speaker_embedding(inputs["speaker_mask"])
        if not (hasattr(self, "exemplar_tokenizer") and hasattr(self, "exemplar_model")):
            out = self.base_model(
                # input_ids=inputs["input_ids"],
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                labels=labels["input_ids"]
            )
        else:
            encoder_output = self.base_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"]
                )
            hidden_states = encoder_output[0]
            exemplar_representations = self._exemplar_mean_pool_representation(exemplars)
            hidden_states = torch.cat([hidden_states,
                exemplar_representations.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)],
                dim=2)
            decoder_input = self.transform_to_t5_decoder(hidden_states)
            out = self.base_model(
                    encoder_outputs=[decoder_input],
                    attention_mask=inputs["attention_mask"],
                    labels=labels["input_ids"]
                    )
        return out, labels

    def generate(self, context, exemplars=None, mode="topk"):
        inputs = self._tokenize_input(context, max_length=self.max_source_length, padding=True, truncation=True)
        inputs_embeds = self.base_model.encoder.embed_tokens(inputs["input_ids"]) + self.speaker_embedding(inputs["speaker_mask"])
        if not (hasattr(self, "exemplar_tokenizer") and hasattr(self, "exemplar_model")):
            if mode == "topk":
                with torch.no_grad():
                    generated = self.base_model.generate(
                                               input_ids=inputs["input_ids"],
                                               inputs_embeds=inputs_embeds,
                                               attention_mask=inputs["attention_mask"],
                                               max_length=20,
                                               do_sample=True,
                                               top_k=20,
                                               temperature=0.9,
                                               early_stopping=True,
                                               num_return_sequences=1)
            elif mode == "beam":
                with torch.no_grad():
                    generated = self.base_model.generate(
                                               input_ids=inputs["input_ids"],
                                               inputs_embeds=inputs_embeds,
                                               attention_mask=inputs["attention_mask"],
                                               max_length=20,
                                               num_beams=8,
                                               early_stopping=True,
                                               num_return_sequences=1)
        else:
            encoder_output = self.base_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"]
                )
            hidden_states = encoder_output[0]
            exemplar_representations = self._exemplar_mean_pool_representation(exemplars)
            hidden_states = torch.cat([hidden_states,
                exemplar_representations.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)],
                dim=2)
            decoder_input = self.transform_to_t5_decoder(hidden_states)
            encoder_output["last_hidden_state"] = decoder_input
            if mode == "topk":
                with torch.no_grad():
                    generated = self.base_model.generate(
                                               input_ids=inputs["input_ids"],
                                               encoder_outputs=encoder_output,
                                               attention_mask=inputs["attention_mask"],
                                               max_length=20,
                                               do_sample=True,
                                               top_k=20,
                                               temperature=0.9,
                                               early_stopping=True,
                                               num_return_sequences=1)
            elif mode == "beam":
                with torch.no_grad():
                    generated = self.base_model.generate(
                                               input_ids=inputs["input_ids"],
                                               encoder_outputs=encoder_output,
                                               attention_mask=inputs["attention_mask"],
                                               max_length=20,
                                               num_beams=8,
                                               early_stopping=True,
                                               num_return_sequences=1)


        hyp = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
              for g in generated]
        return hyp


class T5EncoderClassifier(nn.Module):
    def __init__(self, size, num_labels=2, strategy=0):
        super().__init__()
        
        if size=="base":
            in_features = 768
        elif size=="large":
            in_features = 1024
            
        self.tokenizer = T5Tokenizer.from_pretrained("t5-"+size)
        self.model = T5EncoderModel.from_pretrained("t5-"+size)
        self.classifier = nn.Linear(in_features, num_labels)
        self.strategy = strategy
        
    def forward(self, context, response):        
        max_len = 768
        data = [[x, y] for x, y in zip(context, response)]
        batch = self.tokenizer(data, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(input_ids=batch["input_ids"].to(self.model.device), attention_mask=batch["attention_mask"].to(self.model.device))
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        logits = self.classifier(sequence_output)        
        return logits
    
    def convert_to_probabilities(self, logits):
        if self.strategy == 0:
            probs = F.softmax(logits, 1)
        elif self.strategy == 1:
            probs = F.gumbel_softmax(logits, tau=1, hard=False)
        elif self.strategy == 2:           
            probs = F.gumbel_softmax(logits, tau=1, hard=True)
        return probs
    
    def output_from_logits(self, context, decoded_logits, response_mask):
        '''
        b: batch_size, l: length of sequence, v: vocabulary size, d: embedding dim
        decoded_probabilities -> (b, l, v)
        attention_mask -> (b, l)
        embedding_weights -> (v, d)
        output -> (b, num_labels)
        '''
        # encode context #
        max_len = 768
        batch = self.tokenizer(context, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        context_ids = batch["input_ids"].to(self.model.device)
        context_mask = batch["attention_mask"].to(self.model.device)
        context_embeddings = self.model.encoder.embed_tokens(context_ids)
        
        # encode response #
        decoded_probabilities = self.convert_to_probabilities(decoded_logits)
        embedding_weights = self.model.encoder.embed_tokens.weight
        response_embeddings = torch.einsum("blv, vd->bld", decoded_probabilities, embedding_weights)
        
        # concatenate #
        merged_embeddings = torch.cat([context_embeddings, response_embeddings], 1)
        merged_mask = torch.cat([context_mask, response_mask], 1)        
        outputs = self.model(inputs_embeds=merged_embeddings, attention_mask=merged_mask)
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        logits = self.classifier(sequence_output)
        return logits
    
class T5EncoderRegressor(nn.Module):
    def __init__(self, size, strategy=0):
        super().__init__()
        
        if size=="base":
            in_features = 768
        elif size=="large":
            in_features = 1024
            
        self.tokenizer = T5Tokenizer.from_pretrained("t5-"+size)
        self.model = T5EncoderModel.from_pretrained("t5-"+size)
        self.scorer = nn.Linear(in_features, 1)
        self.strategy = strategy
        
    def forward(self, response):
        max_len = 512
        batch = self.tokenizer(response, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(input_ids=batch["input_ids"].to(self.model.device), attention_mask=batch["attention_mask"].to(self.model.device))        
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        scores = torch.tanh(self.scorer(sequence_output)).flatten()        
        return scores
    
    def convert_to_probabilities(self, logits):
        if self.strategy == 0:
            probs = F.softmax(logits, 1)
        elif self.strategy == 1:
            probs = F.gumbel_softmax(logits, tau=1, hard=False)
        elif self.strategy == 2:           
            probs = F.gumbel_softmax(logits, tau=1, hard=True)
        return probs
    
    def output_from_logits(self, decoded_logits, attention_mask):
        '''
        b: batch_size, l: length of sequence, v: vocabulary size, d: embedding dim
        decoded_probabilities -> (b, l, v)
        attention_mask -> (b, l)
        embedding_weights -> (v, d)
        output -> (b)
        '''
        decoded_probabilities = self.convert_to_probabilities(decoded_logits)
        embedding_weights = self.model.encoder.embed_tokens.weight
        soft_embeddings = torch.einsum("blv, vd->bld", decoded_probabilities, embedding_weights)
        outputs = self.model(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        scores = torch.tanh(self.scorer(sequence_output)).flatten()
        return scores
