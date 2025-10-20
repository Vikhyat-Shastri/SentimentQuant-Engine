"""
Model trainer for custom sentiment models.

Supports fine-tuning pre-trained models on custom datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis training."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (0=negative, 1=neutral, 2=positive)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """Trainer for fine-tuning sentiment models."""
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        num_labels: int = 3,
        device: str = "auto"
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: Pre-trained model to fine-tune
            num_labels: Number of classes (2 or 3)
            device: Device to train on
        """
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing trainer on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        logger.info(f"Model loaded: {model_name}")
    
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        batch_size: int = 16,
        max_length: int = 128
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Batch size
            max_length: Max sequence length
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create training dataset
        train_dataset = SentimentDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create validation dataset
        val_loader = None
        if val_texts and val_labels:
            val_dataset = SentimentDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                max_length
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        output_dir: str = "models/custom"
    ) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            output_dir: Directory to save model
        
        Returns:
            Training history
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader,
                optimizer,
                scheduler
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch + 1}/{epochs} - "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(output_path / "best_model")
                    logger.info(f"New best model saved (val_acc: {val_acc:.4f})")
        
        # Save final model
        self.save_model(output_path / "final_model")
        
        # Save training history
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training complete!")
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path: Path):
        """Save model and tokenizer."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {path}")


def create_training_data_from_csv(
    csv_path: str,
    text_column: str = "text",
    label_column: str = "sentiment"
) -> Tuple[List[str], List[int]]:
    """
    Load training data from CSV.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
    
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)
    
    texts = df[text_column].tolist()
    
    # Convert labels to integers
    # Supports: negative/neutral/positive or 0/1/2
    labels = []
    for label in df[label_column]:
        if isinstance(label, str):
            label_lower = label.lower()
            if 'negative' in label_lower:
                labels.append(0)
            elif 'neutral' in label_lower:
                labels.append(1)
            elif 'positive' in label_lower:
                labels.append(2)
            else:
                labels.append(1)  # Default to neutral
        else:
            labels.append(int(label))
    
    return texts, labels
