import fire
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from torchvision.transforms import ToTensor
from datasets import load_dataset

from training.get_model import get_model
from training.metrics import Metrics


def transforms_data(examples):
    # 画像データを数値データに変換
    examples["pixel_values"] = [image.convert("L") for image in examples["image"]]
    return examples


def collate_fn(examples):
    images = []
    labels = []
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for example in examples:
        # resize to 224x224
        img = example["pixel_values"].convert("RGB")
        img = img.resize((224, 224))
        images.append(ToTensor()(img).to(device_type))
        labels.append(example["labels"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels.unsqueeze(1)}

def run(
        run_name='debug',
        encoder_name='facebook/deit-tiny-patch16-224',
        decoder_name='cl-tohoku/bert-base-japanese-char-v2',
        max_len=300,
        num_decoder_layers=2,
        batch_size=64,
        num_epochs=8,
        fp16=True,
):
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = fp16 and device_type == 'cuda'

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    dataset = load_dataset("mnist")
    dataset = dataset.rename_columns({"label": "labels", "image": "pixel_values"})

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=4,
        output_dir="output",
        logging_steps=10,
        # save_steps=20000,
        save_steps=100,
        # eval_steps=20000,
        eval_steps=100,
        num_train_epochs=num_epochs,
        run_name=run_name
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=dataset['train'].shuffle(seed=42).select(range(2000)),
        eval_dataset=dataset['test'].shuffle(seed=42).select(range(500)),
        data_collator=collate_fn,
    )
    trainer.train()


if __name__ == '__main__':
    fire.Fire(run)
