# Importing the torch library, which is the backbone of many machine learning operations in Python
import torch

# Importing the pipeline function from the transformers package
from transformers import pipeline

# Creating an Automatic Speech Recognition (ASR) pipeline.
# This pipeline uses a model from Facebook's 'wav2vec2' collection, which is designed for speech recognition tasks.
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Importing load_dataset and Audio from the datasets library.
# These functions are used to load and process audio datasets.
from datasets import load_dataset, Audio

# Loading a dataset named 'minds14' from the PolyAI source, specifically the English (US) version for training.
# This dataset is expected to contain audio files and their transcriptions.
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# Casting the 'audio' column of the dataset to match the sampling rate of the speech recognizer's feature extractor.
# This ensures that the audio data is compatible with the speech recognition model.
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# Processing the first four audio files in the dataset through the speech recognition pipeline.
# This will convert the spoken words in the audio files into text.
result = speech_recognizer(dataset[:4]["audio"])

# Printing the transcribed text from each audio file.
# The result contains a list of dictionaries, each with a key 'text' holding the transcribed text.
print([d["text"] for d in result])
