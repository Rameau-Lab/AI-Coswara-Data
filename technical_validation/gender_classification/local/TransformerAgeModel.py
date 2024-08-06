from transformers import Wav2Vec2Model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

class TransformerAgeModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout_rate):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.act(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.output_fc(x)
        return x

    def learn(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.to(device)
        loss_fn = nn.MSELoss()  # Since age prediction is a regression problem

        self.train()

        prev_loss = 9999999
        patience = 20
        countdown = patience

        for epoch in tqdm(range(500)):
            total_loss = 0
            total = 0

            if countdown <= 0 or prev_loss < 1:
                print("Early stopping triggered.")
                print("Best Loss", prev_loss)
                break

            for i, batch in enumerate(train_loader):
                features = batch["feature"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = self.forward(features)
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

        self.eval()
        with torch.no_grad():
            for loader, context in [(val_loader, "Validation"), (test_loader, "Test")]:
                total_loss = 0
                total_squared_error = 0
                total_absolute_error = 0
                count = 0
                sample_count = 0

                for batch in loader:
                    features = batch["feature"].to(device)
                    labels = batch["label"].to(device)

                    outputs = self.forward(features)
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
