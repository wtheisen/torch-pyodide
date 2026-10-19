import time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
X = torch.randn(500, 1, 8, 8); y = torch.tensor([i % 10 for i in range(500)])
m = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128, 10))
opt = torch.optim.Adam(m.parameters(), lr=0.01)
t = time.time()
for ep in range(5):
    for i in range(0, 500, 50):
        opt.zero_grad(); loss = F.cross_entropy(m(X[i:i+50]), y[i:i+50]); loss.backward(); opt.step()
print('cnn 5 epochs x 10 batches: %.2fs loss %.3f' % (time.time() - t, float(loss)))
layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
x = torch.randn(4, 16, 32); mask = nn.Transformer.generate_square_subsequent_mask(16)
opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
t = time.time()
for _ in range(20):
    opt.zero_grad(); layer(x, src_mask=mask).pow(2).mean().backward(); opt.step()
print('transformer block 20 steps: %.2fs' % (time.time() - t))
