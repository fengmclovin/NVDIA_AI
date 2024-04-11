# pip install nemo-toolkit[all]

import nemo
import nemo.collections.asr as nemo_asr

# Initialize the NeMo ASR model
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Define audio file paths for recognition
audio_file_paths = ["path/to/audio_file1.wav", "path/to/audio_file2.wav"]

# Perform speech recognition on each audio file
for audio_file in audio_file_paths:
    # Perform inference and get transcript
    transcript = quartznet.transcribe([audio_file])
    
    # Print the transcript
    print("Audio File:", audio_file)
    print("Transcript:", transcript[0])
