import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as 

                                 

def masked_ce(y_pred, y_true, mask):

    loss = F.cross_entropy(y_pred, y_true, reduction="none")

    loss = loss * mask

    return loss.sum() / (mask.sum() + 1e-8)


def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):

    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc



class Recommender(pl.LightningModule):
    def __init__(
        self,
        out = len(mapping)+2,
        channels=EMBEDDING_LENGTH,
        dropout=0.2,
        lr=1e-4,
        word2vec = w2v
    ):
        super().__init__()
        
        self.lr = lr
        self.dropout = dropout
        self.out = out
        
        self.item_embeddings = w2v

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout, batch_first = True
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=10)

        self.linear_out = Linear(channels, self.out)

        self.do = nn.Dropout(p=self.dropout)


    def forward(self, input_items, mask):

        mask = mask.squeeze(-1)
        
        if torch.cuda.is_available():
            input_items = input_items.to("cuda:0")
            mask = mask.to("cuda:0")
        
        
        input_items = self.encoder(input_items, src_key_padding_mask =mask)
        
        out = self.linear_out(input_items)

        return out
    


    def training_step(self, batch, batch_idx):
        
        
        input_items, y_true, mask, input_tokens_orig = batch

        y_pred = self(input_items, mask)
        

        if torch.cuda.is_available():
            y_pred = y_pred.to("cuda:0")
            y_true = y_true.to("cuda:0")
        
        
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        
        
        input_tokens_orig = input_tokens_orig.view(-1)
        loss_mask = input_tokens_orig == 1
        
        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=loss_mask)
       
        #accuracy = self.masked_accuracy(y_pred=y_pred, y_true=y_true, mask=loss_mask)


        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("train_accuracy", accuracy)

        return loss   
    


    def validation_step(self, batch, batch_idx):
        input_items, y_true, mask, input_tokens_orig = batch

        y_pred = self(input_items, mask)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        input_tokens_orig = input_tokens_orig.view(-1)
        loss_mask = input_tokens_orig == 1
        
        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=loss_mask)   
        
        #accuracy = self.masked_accuracy(y_pred=y_pred, y_true=y_true, mask=loss_mask)
        
        self.log('valid_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("valid_accuracy", accuracy)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }