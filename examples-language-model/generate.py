import torch
from preprocessing import Corpus


class SentenceGenerator:
    def __init__(self, data_path = './wikitext-2', checkpoint_path = './model.pt'):
        self.corpus = Corpus(data_path)
        self.n_tokens = len(self.corpus.dictionary)
        self.sos_token = self.corpus.dictionary.word2idx['<sos>']

        self.transformer = torch.load(checkpoint_path)
        self.transformer.eval()

    def generate(self, n_sentences = 5, sentence_length = 10, temperature = 1.):
        """Generate new sentences using Language Model

        Parameters
        ----------
        sentence_length : int
            number of words to generate
        temperature : float
            higher will increase diversity
        """

        inputs = torch.randint(self.n_tokens, (n_sentences, 1))

        for _ in range(sentence_length):
            prob = self.transformer(inputs)
            prob = torch.exp(prob / temperature)[:, -1, :]
            next_word = torch.multinomial(prob, 1)
            inputs = torch.cat((inputs, next_word), dim = -1)

        return inputs

    def _convert(self, sentence):
        return list(map(lambda x: self.corpus.dictionary.idx2word[x], sentence))

    def convert(self, results):
        results = results.tolist()
        sentences = list(map(self._convert, results))
        
        return sentences

    def run(self, n_sentences = 5, sentence_length = 10, temperature = 1.):
        results = self.generate(n_sentences, sentence_length, temperature)
        sentences = self.convert(results)

        for i, sentence in enumerate(sentences):
            print(f'({i}) {" ".join(sentence)}\n')

if __name__ == '__main__':
    gen = SentenceGenerator(data_path = './wikitext-2', checkpoint_path = './model.pt')
    gen.run(n_sentences = 5, sentence_length = 10, temperature = 1.)