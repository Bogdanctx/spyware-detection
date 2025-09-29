import torch
import matplotlib.pyplot as plt

from src.judge import Judge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset

class ByteDataset(Dataset):
    def __init__(self, file, train = False):
        self.tensors, self.labels = torch.load(file)
        self.train = train
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tensor = self.tensors[index]
        label = self.labels[index]
        
        return tensor, label


class JudgeByteCNN(torch.nn.Module, Judge):
    CONFIG = {
        "VIEW_LEN": 30000, # Number of bytes to consider from each file
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "EPOCHS": 500,
        "BATCH_SIZE": 600
    }

    def __init__(self):
        torch.nn.Module.__init__(self)
        Judge.__init__(self, feature_list=[])

        self.embedding = torch.nn.Embedding(256, 16) # Embedding layer for byte values (0-255)
        self.conv_layers = torch.nn.Sequential(
            # First block
            torch.nn.Conv1d(16, 32, kernel_size = 9, stride = 2, padding = 3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(0.35),
            torch.nn.MaxPool1d(2),
            
            # Second block
            torch.nn.Conv1d(32, 64, kernel_size = 5, stride = 2, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(0.4),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(0.45),
            torch.nn.MaxPool1d(2),
            
            torch.nn.AdaptiveAvgPool1d(128),  # Fixed size output
        )
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 2)
        )

        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing = 0.15, weight=torch.tensor([0.7, 1.3]))
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []

        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def load_data(self):
        """Load train, validation, and test datasets."""
        train_dataset = ByteDataset("./data/train.pt", train = True)
        validation_dataset = ByteDataset("./data/validation.pt")
        test_dataset = ByteDataset("./data/test.pt")

        self.train_loader = DataLoader(train_dataset, 
                                  batch_size=self.CONFIG["BATCH_SIZE"], 
                                  shuffle = True)
        self.validation_loader = DataLoader(validation_dataset, 
                                       batch_size=self.CONFIG["BATCH_SIZE"], 
                                       shuffle = False)
        self.test_loader = DataLoader(test_dataset, 
                                 batch_size=self.CONFIG["BATCH_SIZE"], 
                                 shuffle = False)


    def fit(self): # Override Judge.fit() implementation
        import time

        self.to(self.CONFIG["DEVICE"])
        optimizer = torch.optim.AdamW(params = self.parameters(), 
                                      lr = 0.001,
                                      weight_decay = 0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                               mode = 'min',
                                                               factor = 0.5,
                                                               patience = 8)
        
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(self.CONFIG["EPOCHS"]):
            self.train()

            print(f"====================== Epoch {epoch + 1}/{self.CONFIG['EPOCHS']} ======================")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]}")
            start = time.time()

            total_loss = 0
            correct = 0
            total = 0

            for X, y in self.train_loader:
                X = X.to(self.CONFIG["DEVICE"])
                y = y.to(self.CONFIG["DEVICE"])

                optimizer.zero_grad() # set gradient to 0
                outputs = self(X)
                loss = self.loss_function(outputs, y)
                loss.backward() # backpropagation
                optimizer.step() # update weights
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            end = time.time()
            print(f"Time taken for epoch: {end - start:.2f} seconds")

            # Evaluate after train
            avg_loss = total_loss / len(self.train_loader)
            train_accuracy = correct / total
            print(f"Training accuracy: {train_accuracy * 100:.2f}% | Training loss: {avg_loss:.4f}")
            self.train_loss.append(avg_loss)
            self.train_accuracy.append(train_accuracy)
            ########### Finished evaluating on training set ###########

            ################ Evaluate on validation set ################

            results = None
            if epoch % 3 == 0:
                results = self.evaluate(self.validation_loader, True)
            else:
                results = self.evaluate(self.validation_loader)

            self.validation_loss.append(results["loss"])
            self.validation_accuracy.append(results["accuracy"])
            if results["loss"] < best_val_loss:
                best_val_loss = results["loss"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 10:
                    print("Early stopping triggered.")
                    break

            print(f"Validation accuracy: {results['accuracy'] * 100:.2f}% | Validation loss: {results['loss']:.4f}")
            #####################################################################

            scheduler.step(results['loss']) # update learning rate
            print("=" * 70)
            print("\n")

    # X should be a tensor
    def predict_proba(self, X):
        self.to(self.CONFIG["DEVICE"])
        with torch.no_grad():
            X = X.to(self.CONFIG["DEVICE"])
            outputs = self(X)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def evaluate(self, dataset_loader, debug_text=False):
        self.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        for X, y in dataset_loader:
            X = X.to(self.CONFIG["DEVICE"])
            y = y.to(self.CONFIG["DEVICE"])

            with torch.no_grad():
                outputs = self(X)
                loss = self.loss_function(outputs, y)
            
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        if debug_text:
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
            print("Classification Report:")
            print(classification_report(all_labels, all_preds))

        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            "accuracy": accuracy,
            "loss": total_loss / len(dataset_loader),
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def save_model(self, model_path):
        """Save the trained model to disk."""
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.CONFIG["DEVICE"]))
        self.eval()
        return self
    
    def plot_metrics(self):
        # Plot training loss
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

        # Plot training accuracy
        plt.plot(self.train_accuracy, label='Training Accuracy')
        plt.plot(self.validation_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    judge = JudgeByteCNN()
    judge.fit()
    judge.evaluate(judge.test_loader, True)
    judge.plot_metrics()
    judge.save_model('./judges/judge_cnn.pth')