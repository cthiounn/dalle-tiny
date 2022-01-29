import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers 
from transformers import BartTokenizer, BartForConditionalGeneration

class TinyDalleDataset(Dataset):
    def __init__(self, csv_file,dataset_type):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        super(TinyDalleDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.dataset_type=dataset_type
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.max_length=255
        self.padding="max_length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name=self.data.iloc[idx,0]
        caption = self.data.iloc[idx, 1]
        inputs=self.tokenizer(caption, return_tensors="pt",max_length=self.max_length,padding=self.padding)
        
        
        return [image_name], inputs['input_ids'].squeeze()