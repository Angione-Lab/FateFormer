import torch
from torch import nn
import math


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Obtain the output and attention weights directly from self.self_attn
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=False,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights
    
class SingleTransformer(nn.Module):
    
    """
    Transformer-based model for each modality.
    Args:
        vocab_size (int): Vocabulary size. (set 1 if projection is used.)
        seq_len (int): Sequence length.
        n_encoder_layers (int): Number of transformer encoder layers.
        n_heads (int): Number of attention heads.
        n_batches (int): Number of batches.
        d_tokens (int): Dimension of the token embeddings.
        d_ff (int): Dimension of the feedforward layer.
        d_batch (int): Dimension of the batch embeddings.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
    Attributes:
        count_embedding (torch.Tensor): Count embeddings.
        id_embeddings (torch.Tensor): ID embeddings.
        batch_embedding (nn.Embedding): Batch embeddings.
        layer_norm (nn.LayerNorm): Layer normalization.
        cls_token (torch.Tensor): CLS token.
        encoder (nn.TransformerEncoder): Transformer encoder.
        mask_output_layer (nn.Linear): Mask output layer.
        cls_attention (nn.MultiheadAttention): Multihead attention for CLS token.
        cls_norm1 (nn.LayerNorm): Layer normalization for CLS token.
        cls_norm2 (nn.LayerNorm): Layer normalization for CLS token.
        cls_ffn (nn.Sequential): Feedforward network for CLS token.
        cls_output_layer (nn.Linear): Output layer for CLS token.
        pretrained (bool): Flag indicating if pretrained weights are frozen.
    Methods:
        forward(x, batch_indices, masked_lm=False, return_attention=False, return_embeddings=False):
            Forward pass of the module.
        freeze_pretrained_weights():
            Freeze the pretrained weights.
        unfreeze_pretrained_weights():
            Unfreeze the pretrained weights.
        create_count_embeddings(max_count, embed_size):
            Create count embeddings.
        get_latent_space(inputs, batch_indices, batch_size=32):
            Get the latent space representation and predictions.
    """
    def __init__(self, model_type, vocab_size, seq_len,
                 n_encoder_layers, n_heads, n_batches,
                 d_model, d_ff,
                 dropout_rate=0.0):
        super(SingleTransformer, self).__init__()

        if model_type not in ['RNA', 'ATAC', 'Flux']:
                raise ValueError("model_type must be one of 'RNA', 'ATAC', 'Flux'")
        
        self.model_type = model_type
        
        if self.model_type == 'RNA':
            self.count_embedding_fix = self.create_count_embeddings(vocab_size, d_model)
        else:
            self.count_embedding_proj = nn.Linear(1, d_model)

        self.id_embeddings = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.id_embeddings, mean=0.0, std=0.02)
        self.batch_embedding = nn.Embedding(n_batches, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.token_layer_norm = nn.LayerNorm(d_model)
        self.batch_layer_norm = nn.LayerNorm(d_model)
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout_rate, batch_first=True)
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        self.mask_output_layer = nn.Linear(d_model, vocab_size)

        self.cls_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.cls_norm1 = nn.LayerNorm(d_model)
        self.cls_norm2 = nn.LayerNorm(d_model)
        self.cls_ffn = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(d_ff, d_model)
                    )
        self.dropout = nn.Dropout(dropout_rate)
        self.cls_output_layer = nn.Linear(d_model, 1)
         
    def forward(self, x, batch_indices, masked_lm=False, return_attention=False, return_embeddings=False, return_flow_attention=False):
        
        # [batch_dim, seq_dim, embed_dim] 

        if self.model_type == 'RNA':
            self.count_embedding_fix = self.count_embedding_fix.to(x.device)
            x = x.long()
            x = self.count_embedding_fix[x]
        else:
            x = x.unsqueeze(-1).float()
            x = self.count_embedding_proj(x)
        
        x = x + self.id_embeddings[:, :x.size(1), :]

        batch_embeddings = self.batch_embedding(batch_indices).unsqueeze(1)#.expand(-1, x.size(1), -1) # repeat for the token dim

        # token_embeddings = self.token_layer_norm(x)
        # batch_embeddings = self.batch_layer_norm(batch_embeddings)
        # x = token_embeddings + batch_embeddings
        # print(batch_embeddings.shape, x.shape)
        # print(torch.max(batch_embeddings.flatten()), torch.max(token_embeddings.flatten()))
        # print(torch.min(batch_embeddings.flatten()), torch.min(token_embeddings.flatten()))
        # print("===")
        x = torch.cat((x, batch_embeddings), dim=1) #x + batch_embeddings #
        
        x = self.layer_norm(x)

        attention_flow = []
        for layer in self.encoder.layers:
            x, attn_weights = layer(x)
            if return_flow_attention:
                attention_flow.append(attn_weights)

        other_tokens = x #self.encoder(x)

        if return_embeddings:
            return other_tokens, attention_flow
            
        if masked_lm:
            # exclude the batch embeddings
            other_tokens = other_tokens[:, :-1, :]
            return self.mask_output_layer(other_tokens)

        cls_token = self.cls_token.expand(x.size(0), -1, -1) # repeat for the batch dim
        attended_cls, attention_weights = self.cls_attention(cls_token, other_tokens, other_tokens, need_weights=True, average_attn_weights=False)  
        attended_cls = attended_cls.squeeze(1)

        cls_output = self.cls_norm1(cls_token.squeeze(1) + self.dropout(attended_cls))
        cls_output = self.cls_norm2(cls_output + self.dropout(self.cls_ffn(cls_output)))
        
        preds = self.cls_output_layer(cls_output)
        preds = torch.sigmoid(preds)

        if return_flow_attention:
            return preds, cls_output, attention_weights, attention_flow
        elif return_attention:
            return preds, cls_output, attention_weights
        else:
            return preds, cls_output
    
    def freeze_pretrained_weights(self):
        for name, param in self.named_parameters():
            if not any(x in name for x in ['cls_attention', 'cls_norm', 'cls_ffn', 'cls_token', 'cls_ff_dim', 'cls_output_layer']):
                param.requires_grad = False
        self.pretrained = True

    def unfreeze_pretrained_weights(self):
        for param in self.parameters():
            param.requires_grad = True
        self.pretrained = False
        
    def create_count_embeddings(self, max_count, embed_size):
        embeddings = torch.zeros(max_count + 1, embed_size)
        for i in range(max_count + 1):
            embeddings[i] = torch.tensor([math.sin(i / (10000 ** (2 * (j // 2) / embed_size))) 
                                          if j % 2 == 0 else math.cos(i / (10000 ** (2 * (j // 2) / embed_size))) 
                                          for j in range(embed_size)])
        return embeddings

    def get_latent_space(self, inputs, batch_indices, batch_size=32):
        """
        Get the latent space representation and predictions.
        Args:
            inputs (torch.Tensor): Input tensor.
            batch_indices (torch.Tensor): Batch indices tensor.
            batch_size (int, optional): Batch size. Defaults to 32.
        Returns:
            torch.Tensor: Latent space representation.
            torch.Tensor: Predictions.
        """
        self.eval() 
        latent_space_list, preds_list = [], []
        with torch.no_grad():  
            for i in range(0, inputs.shape[0], batch_size):
                inputs_batch = inputs[i:i + batch_size].float()
                batch_indices_batch = batch_indices[i:i + batch_size].int()
                preds, reduced_dim = self(inputs_batch, batch_indices_batch)
                latent_space_list.append(reduced_dim)
                preds_list.append(preds)
        latent_space = torch.cat(latent_space_list, dim=0)
        preds = torch.cat(preds_list, dim=0)
        return latent_space, preds
    

class MultiModalTransformer(nn.Module):
    def __init__(self, rna_model, atac_model, flux_model, d_model, n_heads_cls, d_ff_cls, dropout_rate=0.0):
        super(MultiModalTransformer, self).__init__()

        self.rna_model = rna_model
        self.atac_model = atac_model
        self.flux_model = flux_model

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # self.modality_embeddings = nn.Embedding(3, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.cls_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads_cls, dropout=dropout_rate, batch_first=True)
        self.cls_norm1 = nn.LayerNorm(d_model)
        self.cls_norm2 = nn.LayerNorm(d_model)
        self.cls_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff_cls),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff_cls, d_model))
        self.cls_output_layer = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, batch_indices, return_attention=False, return_embeddings=False, return_flow_attention=False):
        rna_input, atac_input, flux_input = x[0], x[1], x[2]

        rna_tokens, rna_attention = self.rna_model(rna_input, batch_indices, return_embeddings=True, return_flow_attention=return_flow_attention) # [32, 944, 128]
        atac_tokens, atac_attention = self.atac_model(atac_input, batch_indices, return_embeddings=True, return_flow_attention=return_flow_attention) # [32, 883, 128]
        flux_tokens, flux_attention = self.flux_model(flux_input, batch_indices, return_embeddings=True, return_flow_attention=return_flow_attention) # [32, 168, 128]
        # rna_tokens += self.modality_embeddings(torch.tensor([0]).to(rna_tokens.device))
        # atac_tokens += self.modality_embeddings(torch.tensor([1]).to(atac_tokens.device))
        # flux_tokens += self.modality_embeddings(torch.tensor([2]).to(flux_tokens.device))
        other_tokens = torch.cat((rna_tokens, atac_tokens, flux_tokens), dim=-2) # [32, 1995, 128]

        if return_embeddings:
            return other_tokens
        
        # create mask
        rna_mask = (rna_input.sum(dim=1) != 0).float() # [32]
        # b1 = rna_mask.sum()
        atac_mask = (atac_input.sum(dim=1) != 0).float()  # [32]
        # b2 = atac_mask.sum()
        flux_mask = (flux_input.sum(dim=1) != 0).float() # [32]
        
        rna_mask = rna_mask.unsqueeze(-1).expand(-1, rna_tokens.size(1))  # [32, 944]
        atac_mask = atac_mask.unsqueeze(-1).expand(-1, atac_tokens.size(1))  # [32, 883]
        flux_mask = flux_mask.unsqueeze(-1).expand(-1, flux_tokens.size(1))  # [32, 168]
        other_tokens_mask = torch.cat((rna_mask, atac_mask, flux_mask), dim=1)  # [32, 1995]

        other_tokens = self.layer_norm(other_tokens)
        cls_token = self.cls_token.expand(other_tokens.size(0), -1, -1)  # [32, 1, 128]
        attended_cls, attention_weights = self.cls_attention(cls_token, other_tokens, other_tokens, 
                                                             key_padding_mask=(1 - other_tokens_mask).bool(),
                                                             need_weights=True, average_attn_weights=False)

        attended_cls = attended_cls.squeeze(1)
        cls_output = self.cls_norm1(cls_token.squeeze(1) + self.dropout(attended_cls))
        cls_output = self.cls_norm2(cls_output + self.dropout(self.cls_ffn(cls_output)))

        preds = self.cls_output_layer(cls_output)

        preds = torch.sigmoid(preds)

        if return_flow_attention:
            return preds, cls_output, {
            'rna': rna_attention,
            'atac': atac_attention,
            'flux': flux_attention,
            'cls': attention_weights
        }
        elif return_attention:
            return preds, cls_output, attention_weights
        else:
            return preds, cls_output

    def freeze_pretrained_weights(self):
        self.rna_model.freeze_pretrained_weights()
        self.atac_model.freeze_pretrained_weights()
        self.flux_model.freeze_pretrained_weights()
        for name, param in self.named_parameters():
            if not any(x in name for x in ['cls_attention', 'cls_norm', 'cls_ffn', 'cls_token', 'cls_output_layer']):
                param.requires_grad = False

    def unfreeze_pretrained_weights(self):
        self.rna_model.unfreeze_pretrained_weights()
        self.atac_model.unfreeze_pretrained_weights()
        self.flux_model.unfreeze_pretrained_weights()
        for param in self.parameters():
            param.requires_grad = True

    def get_latent_space(self, X, batch_indices, batch_size=32):
        self.eval() 
        latent_space_list, preds_list = [], []
        rna_input, atac_input, flux_input = X[0], X[1], X[2]
        with torch.no_grad():  
            for i in range(0, rna_input.shape[0], batch_size):
                rna_input_batch = rna_input[i:i + batch_size].float()
                atac_input_batch = atac_input[i:i + batch_size].float()
                flux_input_batch = flux_input[i:i + batch_size].float()
                batch_indices_batch = batch_indices[i:i + batch_size].int()
                preds, reduced_dim = self((rna_input_batch, atac_input_batch, flux_input_batch), batch_indices_batch)
                latent_space_list.append(reduced_dim)
                preds_list.append(preds)
        latent_space = torch.cat(latent_space_list, dim=0)
        preds = torch.cat(preds_list, dim=0)
        return latent_space, preds
    

if __name__=='__main__':
    model = SingleTransformer(model_type='ATAC', vocab_size=1, seq_len=883, n_encoder_layers=2, n_heads=2, n_batches=3, d_tokens=508, d_ff=128, d_batch=4)
    x = torch.rand(32, 883)
    batch_indices = torch.randint(1, 3, (32,))
    print(model(x, batch_indices, masked_lm=True).shape)
    print(model(x, batch_indices, return_attention=True)[0].shape)
    print(model(x, batch_indices, return_embeddings=True).shape)
    print(model(x, batch_indices).shape)
