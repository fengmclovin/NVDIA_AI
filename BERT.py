from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define a text example for classification
text = "This is an example text for sentiment analysis."

# Tokenize the text and convert it into input features for the model
inputs = tokenizer(text, return_tensors='pt')

# Perform inference with the model
outputs = model(**inputs)

# Get the predicted class (label) from the model's output
predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Print the predicted class
print("Predicted class:", predicted_class)
