import os
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
        self.model_checkpoint = "facebook/deit-tiny-patch16-224"
        self.data_dir = Path(dataset_download_path if dataset_download_path else "resources")
        self.test_image_path = os.path.join(self.data_dir, "test", "test1", "1.jpg") #Path("resources\test\test1\1.jpg")
        self.batch_size = 256
        self.num_train_epochs = 5
        self.learning_rate = 5e-3
        self.gradient_accumulation_steps = 4
        self.lora_r = 16
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.target_modules = ["query", "value"]
        self.modules_to_save = ["classifier"]
        self.test_checkpoint_path = Path("deit-tiny-patch16-224-finetuned-lora-cats-vs-dogs\checkpoint-110")

        # Make dirs if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def print_sys_info(self):
        '''
        Just print some system information, helpful to see if gpu is being used or not, especially useful if 
        working on a windows machine.
        Also return a boolean indicating if cuda is available or not. 
        '''
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print(f"Transformers version: {transformers.__version__}")
        print(f"Accelerate version: {accelerate.__version__}")
        print(f"PEFT version: {peft.__version__}")
        return True if torch.cuda.is_available() else False
    
    def make_dataset(self):
        '''
        Basically calls kaggle api to download the cats-vs-dogs dataset and converts it into a hf dataset.
        Also returns a boolean indicating if there were any errors during download/extraction.
        '''
        # download competition data
        error_occurred = False
        try:
            kaggle.api.competition_download_files(competition="dogs-vs-cats", path=self.data_dir, force=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            error_occurred = True
            return error_occurred, e

        zip_path = os.path.join(self.data_dir,"dogs-vs-cats.zip")
        # extract the main zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        except FileNotFoundError:
            print(f"{zip_path} not found. Make sure the file exists.")
            error_occurred = True
            return error_occurred, e

        # extract train and test zip files into train and test folders
        try:
            with zipfile.ZipFile(os.path.join(self.data_dir,"train.zip"),"r") as z:
                z.extractall(os.path.join(self.data_dir, "train"))
                
            with zipfile.ZipFile(os.path.join(self.data_dir,"test1.zip"),"r") as z:
                z.extractall(os.path.join(self.data_dir,"test1.zip"), "test")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            error_occurred = True
            return error_occurred, e
        except Exception as e:
            print(f"An error occurred while extracting: {e}")
            error_occurred = True
            return error_occurred, e

        if not error_occurred:
            print("Data downloaded and extracted successfully.")
            os.remove(zip_path)
            os.remove(os.path.join(self.data_dir,"train.zip"))
            os.remove(os.path.join(self.data_dir,"test1.zip")) 
    
        # make folder structure, do that we can use hugginface datasets library directly
        os.path.join(self.data_dir, "train", "cats").mkdir(parents=True, exist_ok=True)
        os.path.join(self.data_dir, "train", "dogs").mkdir(parents=True, exist_ok=True)
        os.path.join(self.data_dir, "test", "cats").mkdir(parents=True, exist_ok=True)
        os.path.join(self.data_dir, "test", "dogs").mkdir(parents=True, exist_ok=True)
        for file_name in os.listdir(os.path.join(self.data_dir,"train","train")):
            file_path = os.path.join(self.data_dir,"train","train",file_name)
            if file_name.startswith("cat"):
                os.rename(file_path, os.path.join(self.data_dir,"train","cats",file_name))
            elif file_name.startswith("dog"):
                os.rename(file_path, os.path.join(self.data_dir,"train","dogs",file_name))

        # remove now empty folders
        os.rmdir(os.path.join(self.data_dir,"train","train"))
        print("Folder structure created and files moved successfully.")
        return error_occurred, None





dataset = load_dataset("imagefolder", data_dir="resources/train") 



labels = dataset['train'].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


model_checkpoint = "facebook/deit-tiny-patch16-224"




image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)




normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


splits = dataset['train'].train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]


train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )




model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


print_trainable_parameters(model)
# "trainable params: 85876325 || all params: 85876325 || trainable%: 100.00"




config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
# "trainable params: 667493 || all params: 86466149 || trainable%: 0.77"





model_name = model_checkpoint.split("/")[-1]
batch_size = 256

args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
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


trainer.evaluate(val_ds)




config = PeftConfig.from_pretrained("deit-tiny-patch16-224-finetuned-lora-food101\checkpoint-110")
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, "deit-tiny-patch16-224-finetuned-lora-food101\checkpoint-110")


image_processor = AutoImageProcessor.from_pretrained("deit-tiny-patch16-224-finetuned-lora-food101\checkpoint-110")



image = Image.open(r"resources\test\test1\1.jpg")
# image


encoding = image_processor(image.convert("RGB"), return_tensors="pt")


with torch.no_grad():
    outputs = inference_model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", inference_model.config.id2label[predicted_class_idx])




