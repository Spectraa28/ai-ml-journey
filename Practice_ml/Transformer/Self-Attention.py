import numpy as np

def softmax(x):
    shifted = x - np.max(x,axis=-1,keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x,axis=-1,keepdims=True)

def scaled_dot_product_attention(Q,K,V):
    dk = Q.shape[-1]
    scores =  Q @ K.T /np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output , weights

class SelfAttention:
    def __init__(self,d_model,dk,dv,seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model  + dk))
        self.Wq  = rng.normal(0,scale,(d_model,dk))
        self.Wk = rng.normal(0,scale, (d_model,dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0,scale_v,(d_model,dv))
        self.dk = dk
    
    def forward(self,X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output , weights = scaled_dot_product_attention(Q,K,V)
        return output, weights
    
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model,dk,dv,seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
    
    
import torch 
import torch.nn as nn

d_model=8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model , num_heads=n_heads,batch_first=True)

X_torch = torch.randn(1,seq_len, d_model)

output , attn_weights = mha(X_torch,X_torch,X_torch)


print(f"Input shape: {X_torch}")
print()