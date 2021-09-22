import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import Corpus
from language_model import TransformerforLM


class WikiDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        data ; tokenized & concatenated sentences
            = tokenized(['Robert', '<unk>', 'is', 'an', 'English', 'film', '<eos>', 'He', 'had', ...])
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        inputs = self.data[idx : idx + self.seq_len]
        targets = self.data[idx + 1 : idx + self.seq_len + 1]

        return inputs, targets

class Trainer:
    def __init__(self, data_path = './wikitext-2'):
        self.corpus = Corpus(data_path)
        self.n_tokens = len(self.corpus.dictionary)

    def fit(self, 
            n_encoders = 6,
            n_heads = 8,
            d_model = 512,
            d_hidden_ffnn = 2048,
            iterations = 100,
            batch_size = 128,
            seq_len = 35,
            max_len = 5000,
            num_workers = 8,
            shuffle = False,
            evaluation_step = 10):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(1)

        train = WikiDataset(data = self.corpus.train, seq_len = seq_len)
        train_loder = DataLoader(dataset = train,
                                 batch_size = batch_size,
                                 num_workers = num_workers,
                                 shuffle = shuffle)
        
        transformer = TransformerforLM(n_tokens = self.n_tokens,
                                        n_encoders = n_encoders,
                                        n_heads = n_heads,
                                        d_model = d_model,
                                        max_len = max_len,
                                        d_hidden_ffnn = d_hidden_ffnn)

        criterion = nn.NLLLoss()

        transformer.to(device)
        transformer.train()

        optimizer = torch.optim.Adam(transformer.parameters(), lr = 0.05)

        for ep in range(iterations):
            for i, (inputs, targets) in enumerate(train_loder):
                output = transformer(inputs).view(-1, self.n_tokens)
                
                loss = criterion(output.view(-1, self.n_tokens), targets.flatten())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if i%evaluation_step == 0:
                    print(f'{ep}-th epoch / {i}-th batch / loss : {loss.item():.5f}')
                    torch.save(transformer.state_dict(), './model.pt')

        self.transformer = transformer
        self.transformer.eval()

if __name__ == '__main__':
    trainer = Trainer(data_path = './wikitext-2')
    trainer.fit(n_encoders = 2,
                n_heads = 2,
                d_model = 256,
                d_hidden_ffnn = 2048,
                iterations = 5,
                batch_size = 32,
                seq_len = 35,
                max_len = 5000,
                num_workers = 8,
                shuffle = True,
                evaluation_step = 10)