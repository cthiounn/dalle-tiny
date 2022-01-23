import torch
from dalle_tiny.model import TinyDalleModel
from dalle_tiny.util import TinyDalleDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    
    training_data = TinyDalleDataset(csv_file="train.csv",dataset_type="train")
    test_data = TinyDalleDataset(csv_file="validation.csv",dataset_type="val")

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = TinyDalleModel(hidden_size=1,output_size=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        total_loss=0
        for batch in train_dataloader:
            image,label =batch
            predictions=model(label)
            loss =F.cross_entropy(predictions,image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        print("epoch:",epoch,"total_loss:",total_loss
    




if __name__ == "__main__":
    main()
