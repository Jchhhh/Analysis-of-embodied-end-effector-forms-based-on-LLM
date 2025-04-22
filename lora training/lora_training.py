import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

class GripperDataset(Dataset):
    def __init__(self, image_dir, csv_path, processor=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image name']
        description = self.df.iloc[idx]['Structural description']
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.processor:
            inputs = self.processor(
                text=description,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return inputs
        
        return image, description

class FLUXLoRAModel(nn.Module):
    def __init__(self, base_model, lora_config):
        super(FLUXLoRAModel, self).__init__()
        self.model = get_peft_model(base_model, lora_config)
        
    def forward(self, inputs):
        return self.model(**inputs)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, output_dir):
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss = outputs.loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        
        # Save loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_dir, f'loss_epoch_{epoch+1}.png'))
        plt.close()
        
        # Save model checkpoint
        model.save_pretrained(os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}'))

def main():
    # Configuration
    image_dir = 'path_to_images'  # Replace with your image directory
    csv_path = 'test1.csv'  # Converted from Excel
    output_dir = 'training_output'
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    base_model = CLIPModel.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    
    # Initialize model
    model = FLUXLoRAModel(base_model, lora_config).to(device)
    
    # Load dataset
    dataset = GripperDataset(image_dir, csv_path, processor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    train_model(model, train_loader, None, optimizer, num_epochs, device, output_dir)

if __name__ == '__main__':
    main() 