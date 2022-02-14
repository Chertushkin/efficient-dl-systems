import torch
import torch.nn as nn

embedding = nn.Embedding(1024,32).to('cuda')
# 1024 > 1023 (largest index in the created embedding layer)
input = torch.full((1,1),1024,dtype=torch.long, device='cuda')

# out-of-bounds access
embedding_for_index = embedding(input)

result = torch.sigmoid(embedding_for_index)
loss = result.sum()
print(loss.item())
