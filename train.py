import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from engine.wrapper import LitSeg
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


print("cuda:", torch.cuda.is_available())

# Create tokenizer once
tokenizer = AutoTokenizer.from_pretrained(args["bert_type"], use_fast=True) if args.get("use_text", False) else None

# Train/Valid dataset
ds_train = Dataset(
    args["train_csv_path"], args["train_root_path"],
    mode="train",
    image_size=tuple(args["image_size"]),
    tokenizer=tokenizer,
    max_text_len=args.get("max_text_len", 24)
)

ds_valid = Dataset(
    args["train_csv_path"], args["train_root_path"],
    mode="valid",
    image_size=tuple(args["image_size"]),
    tokenizer=tokenizer,
    max_text_len=args.get("max_text_len", 24)
)

dl_train = DataLoader(ds_train, batch_size=args["train_batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
dl_valid = DataLoader(ds_valid, batch_size=args["valid_batch_size"], shuffle=False, num_workers=0, pin_memory=True)

lit_model = LitSeg(args)

ckpt_cb = ModelCheckpoint(
    dirpath=args["model_save_path"],
    filename=args["model_save_filename"],
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    verbose=True,
)

es_cb = EarlyStopping(
    monitor="val_loss",
    patience=args["patience"],
    mode="min",
    verbose=True,
)

trainer = pl.Trainer(
    logger=True,
    min_epochs=args["min_epochs"],
    max_epochs=args["max_epochs"],
    accelerator="gpu",
    devices=1,
    callbacks=[ckpt_cb, es_cb],
    enable_progress_bar=True,
)

print("Start training.")
trainer.fit(lit_model, dl_train, dl_valid)
print("Training complete.")
