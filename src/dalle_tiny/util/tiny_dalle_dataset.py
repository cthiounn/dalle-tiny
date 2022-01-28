import os
import io
import requests
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import transformers 
from transformers import BartTokenizer, BartForConditionalGeneration
from dall_e          import map_pixels, unmap_pixels, load_model
import PIL
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class TinyDalleDataset(Dataset):
    def __init__(self,root_dir_images, csv_file,dataset_type):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        super(TinyDalleDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.dataset_type=dataset_type
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.dev="cpu"
        self.enc= load_model("https://cdn.openai.com/dall-e/encoder.pkl", self.dev)
        self.model= BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.root_dir_images=root_dir_images

    def preprocess(self,img):
        s = min(img.size)
        target_image_size = 256
        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return map_pixels(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name=self.data.iloc[idx,0]
        caption = self.data.iloc[idx, 1]
        inputs=self.tokenizer(caption, return_tensors="pt")
        img=PIL.Image.open(self.root_dir_images+image_name)
        z_logits=self.enc(self.preprocess(img))
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float()
        
        
        sample = {'image': z, 'caption': self.model(**inputs)}
        return sample