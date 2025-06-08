import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader


import torch
from torch import nn
import datasets as ds

from torchtext.vocab import build_vocab_from_iterator, GloVe


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset Wrapper
# ----------------------------
class SST2Dataset(Dataset):
    def __init__(self, token_ids, labels):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


class FasterDeepRnnClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_layers):
        """

        :param embed_dim:
        :param hidden_dim:
        :param rnn_layers:
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_stack = nn.RNN(embed_dim, hidden_dim, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)

        #self.logsoftmax = nn.LogSoftmax(dim=0)
    def forward(self, sentence_tokens):
        all_embeddings = self.embedding(sentence_tokens)
        #all_embeddings = all_embeddings.unsqueeze(1)
        hidden_state_history, _ = self.rnn_stack(all_embeddings)

        feature_extractor_output = hidden_state_history[:, -1, :]
        class_scores = self.linear(feature_extractor_output)
        # logprobs = ...
        return class_scores



def main():
    print("1")


    embed_dim = 50
    model = FasterDeepRnnClassifier(10000, embed_dim, hidden_dim=32, rnn_layers=2)
    model.to(DEVICE)

    for name, tsr in model.named_parameters():
        print(name)


    quit()




    dataset = ds.load_dataset("glue", "sst2")
    train_ds = dataset['train']

    print("1.2")

    # A list of raw sentences
    sentences = train_ds['sentence'][:10000]
    # List of corresponding labels
    labels = train_ds['label'][:10000]

    # How we split sentence into it's "Tokens"
    tokenizer = lambda x: x.split()

    # List of lists of tokens
    tokenized = list(map(tokenizer, sentences))

    print("1.5")

    # Vocab is indexing tokens. so we have Token <=> integer
    vocab = build_vocab_from_iterator(tokenized, specials=["<UNK>"], min_freq=5)
    vocab.set_default_index(0)

    print("2")

    stoi = lambda x: torch.tensor(vocab(x))#, dtype=torch.long)

    integer_tokens = list(map(stoi, tokenized))
    print(integer_tokens[:3])
    #
    # glove_embedder = GloVe(name='6B', dim=50)
    # embedded = glove_embedder.get_vecs_by_tokens(["it", "is", "cool"])

    """
    Looks like we want to lower() all text. another smart tokenization func will be to process it's to it is
    aren't to are not etc, etc...
    """
    #
    # print(embedded.size(), embedded.dtype, embedded.requires_grad)
    # print(embedded)

    print("3")

    train_dataset = SST2Dataset(integer_tokens, labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


    embed_dim = 50
    model = FasterDeepRnnClassifier(len(vocab), embed_dim, hidden_dim=32, rnn_layers=2)
    model.to(DEVICE)

    print(model.named_parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("4")

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs.unsqueeze(0)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


if __name__ == '__main__':
    main()
