"""
Training pipeline for the Siamese Movie Recommender.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.settings import (
    LEARNING_RATE,
    NUM_EPOCHS,
)

from src.model import SiameseNetwork
from src.loss import CosineContrastiveLoss


class Trainer:

    def __init__(
        self,
        model: SiameseNetwork,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        checkpoint_dir: Path,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.criterion = CosineContrastiveLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

        self.best_loss = float("inf")

        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        progress = tqdm(self.train_loader)

        for batch in progress:

            self.optimizer.zero_grad()

            emb_a, emb_b = self.model(

                batch["input_ids_a"].to(self.device),

                batch["attention_mask_a"].to(self.device),

                batch["input_ids_b"].to(self.device),

                batch["attention_mask_b"].to(self.device),

            )

            loss = self.criterion(

                emb_a,

                emb_b,

                batch["label"]
                .to(self.device),

            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_description(

                f"Loss: {loss.item():.4f}"

            )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0

        for batch in self.valid_loader:

            emb_a, emb_b = self.model(

                batch["input_ids_a"].to(self.device),

                batch["attention_mask_a"].to(self.device),

                batch["input_ids_b"].to(self.device),

                batch["attention_mask_b"].to(self.device),

            )

            loss = self.criterion(

                emb_a,

                emb_b,

                batch["label"]
                .to(self.device),

            )

            total_loss += loss.item()

        return total_loss / len(self.valid_loader)

    def save_checkpoint(
        self,
        epoch,
        loss,
    ):

        path = (
            self.checkpoint_dir
            / "best_model.pt"
        )

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "loss": loss,

            },

            path,

        )

    def fit(self):

        for epoch in range(NUM_EPOCHS):

            train_loss = self.train_epoch()

            valid_loss = self.validate()

            print(

                f"Epoch {epoch+1}"

                f" | Train {train_loss:.4f}"

                f" | Valid {valid_loss:.4f}"

            )

            if valid_loss < self.best_loss:

                self.best_loss = valid_loss

                self.save_checkpoint(
                    epoch,
                    valid_loss,
                )