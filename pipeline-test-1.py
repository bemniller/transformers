# Importing the sys module to access system-specific parameters and functions
import sys

# Importing the pipeline function from the transformers package
# This package provides a simple API for performing tasks like Named Entity Recognition (NER)
from transformers import pipeline

# Printing the current Python version and detailed version information
# This is used for debugging and ensuring compatibility
print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)

# Creating a NER pipeline.
# 'ner' refers to Named Entity Recognition, and 'grouped_entities=True' groups together entities that are split into multiple tokens
ner = pipeline("ner", grouped_entities=True)

# Processing a string through the NER pipeline
# The string contains a sample sentence with named entities
results = ner("My name is Ben and I work at PJM in Audubon")

# Iterating over each result in the processed output
# Each result corresponds to a recognized named entity
for result in results:
    # Printing the raw result for each entity for debugging and inspection
    print(result)

    # Formatting and printing the details of each named entity
    # 'entity_group' contains the type of entity (e.g., person, organization)
    # 'word' is the actual word identified as an entity
    # 'score' is the confidence score of the prediction, rounded to 4 decimal places
    print(f"Entity Type: {result['entity_group']}, Word: {result['word']}, with score: {round(result['score'], 4)}")
