import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import Wav2Vec2Model
class SoundDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "feature": torch.tensor(self.features[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)  # Modify dtype as needed
        }

class FT_Wav2Vec():
    def __init__(self):
        self.model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

    def train(self, train_loader, val_loader, test_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        loss_fn = nn.MSELoss()  # Since age prediction is a regression problem

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

        prev_loss = 9999999
        patience = 10
        countdown = patience
        print("starting train")
        for epoch in tqdm(range(500)):
            total_loss = 0
            total = 0

            if countdown <= 0 or prev_loss < 1:
                print("Early stopping triggered.")
                print("Best Loss", prev_loss)
                break

            for i, batch in enumerate(train_loader):
                features = batch["feature"].to(device).squeeze()
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                outputs = self.model(features)
                breakpoint()
                outputs = outputs.squeeze()
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / total
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

            if avg_loss < prev_loss:
                prev_loss = avg_loss
                countdown = patience
            else:
                countdown -= 1

            scheduler.step()
        print("starting eval")
        self.model.eval()
        with torch.no_grad():
            for loader, context in [(val_loader, "Validation"),(test_loader, "Test")]:
                total_loss = 0
                total_squared_error = 0
                total_absolute_error = 0
                count = 0
                sample_count = 0

                for batch in loader:
                    features = batch["feature"].to(device)
                    labels = batch["label"].to(device)

                    outputs = self.model(features)
                    outputs = outputs.squeeze()
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()
                    count += 1

                    # Calculate the squared errors for RMSE
                    squared_errors = (outputs - labels) ** 2
                    total_squared_error += squared_errors.sum().item()
                    sample_count += labels.size(0)

                    # Calculate the absolute errors
                    absolute_errors = torch.abs(outputs - labels)
                    total_absolute_error += absolute_errors.sum().item()

                avg_loss = total_loss / count
                rmse = torch.sqrt(torch.tensor(total_squared_error / sample_count))
                print(f"{context} Average Loss: {avg_loss}")
                print(f"{context} RMSE: {rmse.item()}")

                mae = total_absolute_error / sample_count  # Calculate MAE
                print(f"{context} MAE: {mae}")








