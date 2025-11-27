
import torch
from onebit_asr.conformer import ConformerASR

def test_conformer():
    input_dim = 80
    vocab_size = 100
    model = ConformerASR(input_dim, vocab_size, enc_d_model=32, enc_heads=4, enc_layers=2)
    
    B = 2
    T = 50
    F = input_dim
    feats = torch.randn(B, T, F)
    feat_lens = torch.tensor([50, 40])
    
    batch = {
        'feats': feats,
        'feat_lens': feat_lens
    }
    
    print("Running forward pass...")
    try:
        enc_out, enc_mask, logits = model(batch, precision=32)
        print("Forward pass successful!")
        print("Enc out shape:", enc_out.shape)
        print("Logits shape:", logits.shape)
    except Exception as e:
        print("Forward pass failed!")
        print(e)
        raise e

if __name__ == "__main__":
    test_conformer()
