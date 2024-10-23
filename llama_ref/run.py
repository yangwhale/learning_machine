import torch
import model
import torch_xla2


def main():
  args = model.ModelArgs()
  args.n_layers = 2
  args.vocab_size = 32000

  llama = model.Transformer(args)
  print(llama)

  env = torch_xla2.default_env()

  with env:
    llama.to('jax')
    compiled = torch_xla2.compile(llama)
    tokens = torch.randint(0, 32000, (1, 100))
    freqs_cis = llama.freqs_cis[0: 100]
    seqlen=100
    mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    start_pos = 0
    # https://github.com/pytorch/pytorch/issues/100005
    # torch.triu is buggy when the device is mps: filled values are 
    # nan instead of 0. 
    # When performing key-value caching, we compute the attention scores
    # only for the new sequence. Thus, the matrix of scores is of size
    # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
    # j > cache_len + i, since row i corresponds to token cache_len + i.
    mask = torch.hstack(
        [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
    )
    print(compiled(tokens, 0, freqs_cis, mask))


if __name__ == '__main__':
  main()