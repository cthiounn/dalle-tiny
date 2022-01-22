import os
import io
import requests
import pandas as pd
from torch.utils.data import Dataset, DataLoader, IterableDataset
import transformers 
from transformers import BartTokenizer, BartForConditionalGeneration
from dall_e          import map_pixels, unmap_pixels, load_model



class TinyDalleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(TinyDalleDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.dev="cpu"
        self.enc= load_model("https://cdn.openai.com/dall-e/encoder.pkl", self.dev)
        self.model= BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def download_image(url):
        resp = requests.get(url)
        resp.raise_for_status()
        return PIL.Image.open(io.BytesIO(resp.content))

    def preprocess(img):
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

        # img_name = os.path.join(self.root_dir,
        #                         self.data.iloc[idx, 0])
        # image = io.imread(img_name)
        image_name=self.data.iloc[idx,0]
        image_url="http://images.cocodataset.org/val2017/"+image_name
        caption = self.data.iloc[idx, 1]
        inputs=self.tokenizer(caption, return_tensors="pt")
        sample = {'image': self.enc(download_image(image_url)), 'caption': self.model(**inputs)}

        if self.transform:
            sample = self.transform(sample)

        return sample