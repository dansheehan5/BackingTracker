import librosa
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
import torch
from unet import UNet
from dataset import DSD100Data
from torchaudio import transforms
import matplotlib.pyplot as plt

### Graphs a spectrogram
### Used to visiualize model output
### From https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show()

resample_freq = 16000

batch_size = 8

### Define the transformation
### MelSpectrograms @ 16000 samples
def t(x):
    r = transforms.Resample(441000, resample_freq)
    rs = r(x)
    ms = MelSpectrogram(resample_freq, n_mels=64)
    return ms(rs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

train_data = DSD100Data(test=False, transform=t)
test_data = DSD100Data(test=True, transform=t)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

### DSD100 has 2 channels of audio
m = UNet(2, batch_size)

### Function with 1 epoch of training
def train(model, dl, loss_fn, optimizer):
    ### Send model to GPU for faster training
    model.to(device)
    model.train()
    size = len(train_loader)

    ### Backpropagate and evaluate loss over 1 batch
    for batch, (data, tgt) in enumerate(dl):
        data, tgt = data.to(device), tgt.to(device)
        pred = model(data)
        loss = loss_fn(pred, tgt)

        ### Backpropagate & Update Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(data)
            # plot_spectrogram(data[0][0].cpu(), title="test result", ylabel="mel freq")
            # plot_spectrogram(tgt[0][0].cpu(), title="tgt", ylabel="mel freq")
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

### Tests the model using the supplied loss function and data loader
def test(model, dl, loss_fn):
    num_batches = len(dl)
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, tgt in dl:
            ### Utilize CPU for enhanced performance
            data, tgt = data.to(device), tgt.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, tgt).item()
    test_loss /= num_batches
    print(f"Average loss: {test_loss:>8f}")

loss = torch.nn.MSELoss()
opt = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
train(m, train_loader, loss, opt)

for epoch in range(50):
    train(m, train_loader, loss, opt)

test(m, test_loader, loss)

### Prints out the target and estimated spectrogram of the melody for the first item in the test data loader
for d, t in test_loader:
    d = d.to(device)
    with torch.no_grad():
        pred = m(d)
    spec = pred[0][0].cpu()

    plot_spectrogram(t[0][0], title="tgt")
    plot_spectrogram(spec, title="pred")
    break