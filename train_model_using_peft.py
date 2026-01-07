import os
import shutil
import peft
import torch
import kaggle
import zipfile
import evaluate
import accelerate
import numpy as np
import transformers
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class SetupConfig:
    def __init__(self, dataset_download_path: None):
        self.data_dir = Path(dataset_download_path if dataset_download_path else Path("resources"))
        # Make dirs if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.test_image_path = str(self.data_dir / "test" / "test1" / "1.jpg")
        
    def print_sys_info(self):
        '''
        Just print some system information, helpful to see if gpu is being used or not, especially useful if 
        working on a windows machine.
        Also return a boolean indicating if cuda is available or not. 
        '''
        cuda = torch.cuda.is_available()
        print("CUDA available:", cuda)
        if cuda:
            print("CUDA device count:", torch.cuda.device_count())
            print("Current CUDA device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print(f"Transformers version: {transformers.__version__}")
        print(f"Accelerate version: {accelerate.__version__}")
        print(f"PEFT version: {peft.__version__}")
        return cuda
    
    def make_dataset(self):
        '''
        Basically calls kaggle api to download the cats-vs-dogs dataset and converts it into a hf dataset.
        Also returns a boolean indicating if there were any errors during download/extraction.
        '''
        # download competition data
        error_occurred = False
        try:
            kaggle.api.competition_download_files(competition="dogs-vs-cats", path=str(self.data_dir), force=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            error_occurred = True
            return error_occurred, e

        zip_path = self.data_dir / "dogs-vs-cats.zip"
        # extract the main zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        except FileNotFoundError as e:
            print(f"{zip_path} not found. Make sure the file exists.")
            error_occurred = True
            return error_occurred, e

        train_zip = self.data_dir / "train.zip"
        test_zip  = self.data_dir / "test1.zip"
        # extract train and test zip files into train and test folders
        try:
            with zipfile.ZipFile(train_zip,"r") as z:
                z.extractall(path=self.data_dir / "train")

            out_dir = self.data_dir / "test" / "test1"
            out_dir.mkdir(parents=True, exist_ok=True)    
            with zipfile.ZipFile(test_zip,"r") as z:
                z.extractall(path=self.data_dir / "test")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            error_occurred = True
            return error_occurred, e
        except Exception as e:
            print(f"An error occurred while extracting: {e}")
            error_occurred = True
            return error_occurred, e
    
        # make folder structure, so that we can use hugginface datasets library directly
        cats_dir = self.data_dir / "train" / "cats"
        dogs_dir = self.data_dir / "train" / "dogs"
        cats_dir.mkdir(parents=True, exist_ok=True)
        dogs_dir.mkdir(parents=True, exist_ok=True)

        # train.zip extracted into resources/train/train/*.jpg
        raw_train_dir = self.data_dir / "train" / "train"
        if raw_train_dir.exists():
            try:
                for p in raw_train_dir.iterdir():
                    if not p.is_file():
                        continue
                    name = p.name.lower()
                    if name.startswith("cat"):
                        shutil.move(str(p), str(cats_dir / p.name))
                    elif name.startswith("dog"):
                        shutil.move(str(p), str(dogs_dir / p.name))
                raw_train_dir.rmdir()  # now empty
            except Exception as e:
                print(f"Error organizing train images: {e}")
                return True, e
        else:
            print(f"Note: {raw_train_dir} not found (maybe already organized).")
        # remove now empty folders
        try:
            zip_path.unlink(missing_ok=True)
            train_zip.unlink(missing_ok=True)
            test_zip.unlink(missing_ok=True)
        except Exception:
            pass
        print("Folder structure created and files moved successfully.")
        return error_occurred, None


class DataPipeline:
    def __init__(self, data_dir: str = "resources/train", model_checkpoint: str = "facebook/deit-tiny-patch16-224"):
        self.data_dir = data_dir
        self.model_checkpoint = model_checkpoint
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

    def load_data(self):
        dataset = load_dataset("imagefolder", data_dir=self.data_dir)
        def preprocess_train(example_batch):
            """Apply train_transforms across a batch.(A single image is also handled)"""
            imgs = example_batch["image"]
            if isinstance(imgs, list):
                example_batch["pixel_values"] = [train_transforms(im.convert("RGB")) for im in imgs]
            else:
                example_batch["pixel_values"] = train_transforms(imgs.convert("RGB"))
            return example_batch


        def preprocess_val(example_batch):
            """Apply val_transforms across a batch.(A single image is also handled)"""
            imgs = example_batch["image"]
            if isinstance(imgs, list):
                example_batch["pixel_values"] = [val_transforms(im.convert("RGB")) for im in imgs]
            else:
                example_batch["pixel_values"] = val_transforms(imgs.convert("RGB"))
            return example_batch

        labels = dataset['train'].features["label"].names
        label2id, id2label = dict(), dict()
        
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        train_transforms = Compose(
            [
                RandomResizedCrop(self.image_processor.size["height"]),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

        val_transforms = Compose(
            [
                Resize(self.image_processor.size["height"]),
                CenterCrop(self.image_processor.size["height"]),
                ToTensor(),
                normalize,
            ]
        )
        splits = dataset['train'].train_test_split(test_size=0.1)
        train_ds = splits["train"]
        val_ds = splits["test"]
        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)       
        return train_ds, val_ds, label2id, id2label

class ModelPipeline:
    def __init__(self, rank: int = 16, alpha: int = 16, target_modules: list = ["query", "value"], 
                 lora_dropout: float = 0.1, bias: str = "none", modules_to_save: list = ["classifier"]):
        self.model_checkpoint = "facebook/deit-tiny-patch16-224"
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint, use_fast=True)
        # lora config
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.modules_to_save = modules_to_save

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    def get_lora_model(self, model):
        config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            modules_to_save=self.modules_to_save,
        )
        lora_model = get_peft_model(model, config)
        return lora_model
    
    def get_base_model(self, label2id, id2label):
        model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        return model
    
    def train_model(self, train_ds, val_ds, lora_model, image_processor, label2id, id2label):
        model_name = self.model_checkpoint.split("/")[-1]
        batch_size = 256

        args = TrainingArguments(
            f"{model_name}-finetuned-lora-cats-vs-dogs-101",
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-3,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            num_train_epochs=5,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            label_names=["labels"],
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            """Computes accuracy on a batch of predictions"""
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=predictions, references=eval_pred.label_ids)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        trainer = Trainer(
            lora_model,
            args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        train_results = trainer.train()
        return trainer.evaluate(val_ds), trainer
    
    def model_inference(self, label2id, id2label, 
                        lora_checkpoint_path: str = r"deit-tiny-patch16-224-finetuned-lora-cats-vs-dogs101\checkpoint-110",
                        test_image_path: str = Path("resources\test\test1\1.jpg"),
                        trainer: Trainer = None):
        # Load base model
        if trainer:
            ckpt = trainer.state.best_model_checkpoint
        else:
            ckpt = lora_checkpoint_path
        if not ckpt:
            ckpt = trainer.args.output_dir  # fallback

        peft_config = PeftConfig.from_pretrained(ckpt)
        base_model = AutoModelForImageClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        # Load the LoRA model
        lora_model = PeftModel.from_pretrained(base_model, ckpt)
        image_processor = AutoImageProcessor.from_pretrained(peft_config.base_model_name_or_path, use_fast=True)
        image = Image.open(test_image_path)
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = lora_model(**encoding)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", lora_model.config.id2label[predicted_class_idx])
        return lora_model.config.id2label[predicted_class_idx]

if __name__ == "__main__":
    # Setup configuration
    setup_config = SetupConfig(dataset_download_path=None)
    cuda_available = setup_config.print_sys_info()
    error_occurred, error = setup_config.make_dataset()
    if error_occurred:
        print("Exiting due to error in dataset preparation.")
        exit(1)

    # Data pipeline
    data_pipeline = DataPipeline(data_dir=os.path.join(setup_config.data_dir, "train"))
    train_ds, val_ds, label2id, id2label = data_pipeline.load_data()

    # Model pipeline
    model_pipeline = ModelPipeline()
    base_model = model_pipeline.get_base_model(label2id, id2label)
    lora_model = model_pipeline.get_lora_model(base_model)
    model_pipeline.print_trainable_parameters(lora_model)

    # Train model
    eval_results, trainer = model_pipeline.train_model(train_ds, val_ds, lora_model, data_pipeline.image_processor, label2id, id2label)
    print("Evaluation results:", eval_results)

    # Inference
    predicted_label = model_pipeline.model_inference(label2id, id2label, 
                                                     lora_checkpoint_path="deit-tiny-patch16-224-finetuned-lora-cats-vs-dogs-101/checkpoint-epoch-5",
                                                     test_image_path=setup_config.test_image_path,
                                                     trainer=trainer)
    print("Predicted label for test image:", predicted_label)




