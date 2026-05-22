import os
import numpy as np

# 1. Set working directory
os.chdir("/home/gabriel/cleaned_hifi_vocoder/hifi-gan")


import torch
import json
import soundfile as sf
import matplotlib.pyplot as plt
from models import Generator
from env import AttrDict
from meldataset import mel_spectrogram


# 2. Load config
config_path = "config_yoruba.json"
with open(config_path, "r") as f:
    config = AttrDict(json.load(f))

# 3. Initialize generator + checkpoint
checkpoint_path = "ckpts_yoruba_2/g_00045000"
generator = Generator(config).to("cuda")
checkpoint = torch.load(checkpoint_path, map_location="cuda")
state_dict = checkpoint["generator"] if "generator" in checkpoint else checkpoint["model"]
generator.load_state_dict(state_dict, strict=False)
generator.eval()
generator.remove_weight_norm()

# 4. Load mel spectrogram
mel_path = "/home/gabriel/cleaned_hifi_vocoder/test_mel/orig_4_Audio_UnivOfIbadan_3_PUBLIC_HEALTH_YOR_00004.npy"
mel_array = np.load(mel_path)
mel_example = torch.from_numpy(mel_array).unsqueeze(0).to("cuda")
print(mel_example.shape)

# 5. Run inference
with torch.no_grad():
    audio = generator(mel_example)

# 6. Diagnostics
audio_out = audio.squeeze().cpu().numpy()

# Plot waveform
plt.figure(figsize=(10,4))
plt.plot(audio_out)
plt.title("Generated waveform")
plt.savefig("waveform.png")
plt.close()

# Plot spectrogram of generated audio
# Keep batch dimension intact (no squeeze(1))
# Plot spectrogram of generated audio
# audio has shape [1, 1, T] after generator
audio_for_spec = audio.squeeze(1)  # -> [1, T]

# Now mel_spectrogram returns [1, n_mels, time]
y_hat_spec = mel_spectrogram(audio_for_spec,
                             config.n_fft, config.num_mels,
                             config.sampling_rate, config.hop_size,
                             config.win_size, config.fmin, config.fmax)

print("Generated mel shape:", y_hat_spec.shape)  # should be [1, num_mels, time]

plt.figure(figsize=(10,4))
plt.imshow(y_hat_spec.cpu().numpy(), aspect="auto", origin="lower")
plt.title("Generated spectrogram")
plt.savefig("spectrogram.png")
plt.close()

# 7. Normalize audio before saving
if np.max(np.abs(audio_out)) > 0:
    audio_out = audio_out / np.max(np.abs(audio_out))

sf.write("gen_4_Audio_UnivOfIbadan_3_PUBLIC_HEALTH_YOR_00004.wav", audio_out, samplerate=config.sampling_rate)
print("Audio saved as generated_mel_rescaled.wav")

# compared generated vs ground truth spctrogram

fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].imshow(mel_example[0].cpu().numpy(), aspect="auto", origin="lower")
axs[0].set_title("Ground truth mel")
axs[1].imshow(y_hat_spec.cpu().numpy(), aspect="auto", origin="lower")
axs[1].set_title("Generated mel")
plt.savefig("mel_comparison.png")
plt.close()
