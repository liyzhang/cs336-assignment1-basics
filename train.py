import numpy as np
import torch
from pathlib import Path
from tokenizer import Tokenizer
from train_bpe import load_merges, load_vocab, save_merges, save_vocab, train_bpe
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerLM
from nn_utils import cross_entropy
from adamw import AdamW


class LMDataset(Dataset):
    def __init__(self, tokens: np.array, context_length: int):
        super().__init__()
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length - 1

    def __getitem__(self, index):

        tokens = torch.from_numpy(self.tokens[index : index + self.context_length + 1].astype(np.int64))
        return tokens[:-1], tokens[1:]


def get_batch(
    dataset: np.array, batch_size: int, context_length: int, device: str
) -> tuple[torch.tensor, torch.tensor]:
    """
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    starts = np.random.randint(0, len(dataset) - context_length, batch_size)

    tokens = torch.from_numpy(np.stack([dataset[i : i + context_length + 1] for i in starts]))

    x = tokens[:, :-1].to(device=device, dtype=torch.long)
    y = tokens[:, 1:].to(device=device, dtype=torch.long)

    return x, y


def tokenize(infile_path: str, outfile_path: str, tokenizer: Tokenizer):
    with open(infile_path, "r", encoding="utf-8") as fin:
        text = fin.read()
        token_ids = tokenizer.encode(text=text)
        with open(outfile_path, mode="wb") as fout:
            np.asarray(token_ids, dtype=np.uint16).tofile(fout)

def eval(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    casusal_mask: torch.tensor, # b, l, l
    device: str
): 
    model.eval()
    i, total_loss = 0, 0
    with torch.no_grad():
        for x, y in eval_dataloader:
            logits = model(x.to(device))
            ce = cross_entropy(logits, y.to(device))
            total_loss += ce.item()
            i+=1
    
    model.train()
    return total_loss/i

def train(
    train_dataloader,
    eval_dataloader,
    device,
    batch_size=64,
    max_seq_len=256,
    d_model=512,
    num_heads=4,
    d_ff=1344,
    num_layers=4,
    vocab_size=10000,
    theta=10000,
    lr=3e-4,
    beta1=0.9,
    beta2=0.99,
    eps=1e-5,
    weight_decay=0.01,
):
    # initialize model and optimizer
    model = TransformerLM(d_model, num_heads, d_ff, max_seq_len, num_layers, vocab_size, theta)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    optimizer = AdamW(model.parameters(), lr, (beta1, beta2), eps, weight_decay)

    # train
    causal_mask = (
        torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=bool), diagonal=0).unsqueeze(0).expand(batch_size, -1, -1)
    ).to(device)
    
    i = 0
    for x, y in train_dataloader:
        optimizer.zero_grad()

        logits = model(x.to(device), causal_mask)
        ce = cross_entropy(logits=logits, targets=y.to(device))

        ce.backward()
        optimizer.step()

        print(f"train_step:{i}, ce:{ce.item()}")
        i += 1
        
        if i%100 == 0:
            eval_loss = eval(model, eval_dataloader, causal_mask)
            print(f"train_step:{i}, eval_ce:{eval_loss.item()}")


def main():
    # train bpe, build vocab and merges
    # vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 10000, '<|endoftext|>')
    # save_vocab(vocab, "./data/TinyStoriesV2-GPT4-valid.vocab")
    # save_merges(merges, "./data/TinyStoriesV2-GPT4-valid.merges")

    # build tokenizer
    # merges = load_merges("./data/TinyStoriesV2-GPT4-valid.merges")
    # vocab = load_vocab("./data/TinyStoriesV2-GPT4-valid.vocab")
    # tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens='<|endoftext|>')

    # build train dataset
    train_token_path = "./data/TinyStoriesV2-GPT4-valid.txt.bin"
    # tokenize("./data/TinyStoriesV2-GPT4-valid.txt", train_token_path, tokenizer)

    # build dataset and dataloader
    train_data = np.memmap(train_token_path, dtype=np.uint16, mode="r")
    train_dataset = LMDataset(train_data, 256)
    print(f"train token #: {len(train_dataset):,}")
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(train_dataloader, eval_dataloader, device)


if __name__ == "__main__":
    main()
