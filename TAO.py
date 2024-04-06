import torch
import torchaudio
from transformers import Speech2TextProcessor, SpeechEncoderDecoderModel

# Load the pre-trained Speech-to-Text model and processor
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-small-librispeech-asr")

# Download and load a sample audio file
url = "https://raw.githubusercontent.com/pytorch/audio/master/.github/data/16-122828-0002.wav"
filename = torchaudio.utils.download_url(url, "./sample.wav")
input_audio, _ = torchaudio.load(filename)

# Perform speech recognition
input_features = processor(input_audio, return_tensors="pt")
with torch.no_grad():
    logits = model(input_features.input_features).logits

# Decode the output logits into text
decoded_results = processor.batch_decode(logits, skip_special_tokens=True)
print("Decoded Text:", decoded_results[0])
