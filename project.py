import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import TransformerModel

from sklearn.model_selection import train_test_split
from transformers import M2M100Tokenizer

# Assuming you have a custom dataset class (e.g., TranslationDataset) for training
# and a separate dataset for validation and testing

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        src = batch['source_tokens'].to(device)
        tgt = batch['target_tokens'].to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])  # Exclude the last token for input during training
        loss = criterion(output.view(-1, output.size(-1)), tgt[1:, :].view(-1))  # Exclude the first token for target
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            src = batch['source_tokens'].to(device)
            tgt = batch['target_tokens'].to(device)

            output = model(src, tgt[:-1, :])  # Exclude the last token for input during evaluation
            loss = criterion(output.view(-1, output.size(-1)), tgt[1:, :].view(-1))  # Exclude the first token for target

            total_loss += loss.item()

    return total_loss / len(val_loader)

def translate(model, sentence, max_len, tokenizer, device):
    model.eval()

    # Tokenize and add batch dimension
    input_tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)

    # Generate output sequence
    output_tokens = [tokenizer.bos_token_id]  # Start with the <bos> token
    for _ in range(max_len):
        output_tokens_tensor = torch.tensor(output_tokens).unsqueeze(0).to(device)

        # Model forward pass
        output = model(input_tokens, output_tokens_tensor)

        # Get predicted next token
        predicted_token = torch.argmax(output[:, -1, :], dim=-1).item()

        # Break if <eos> token is predicted
        if predicted_token == tokenizer.eos_token_id:
            break

        # Append predicted token to output sequence
        output_tokens.append(predicted_token)

    # Decode the output sequence
    output_sentence = tokenizer.decode(output_tokens, skip_special_tokens=True)

    return output_sentence

def prepare_data(data):
    # Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Extract Turkish and English sentences from the split data
    train_turkish_sentences, train_english_sentences = zip(*train_data)
    val_turkish_sentences, val_english_sentences = zip(*val_data)
    test_turkish_sentences, test_english_sentences = zip(*test_data)

    # Tokenize the sentences using a pre-trained tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    # Tokenize and encode Turkish sentences
    train_turkish_tokens = tokenizer(train_turkish_sentences, return_tensors="pt", padding=True, truncation=True)
    val_turkish_tokens = tokenizer(val_turkish_sentences, return_tensors="pt", padding=True, truncation=True)
    test_turkish_tokens = tokenizer(test_turkish_sentences, return_tensors="pt", padding=True, truncation=True)

    # Tokenize and encode English sentences
    train_english_tokens = tokenizer(train_english_sentences, return_tensors="pt", padding=True, truncation=True)
    val_english_tokens = tokenizer(val_english_sentences, return_tensors="pt", padding=True, truncation=True)
    test_english_tokens = tokenizer(test_english_sentences, return_tensors="pt", padding=True, truncation=True)

    # Convert the tokenized sentences to PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(
        train_turkish_tokens["input_ids"], train_turkish_tokens["attention_mask"],
        train_english_tokens["input_ids"], train_english_tokens["attention_mask"]
    )

    val_dataset = torch.utils.data.TensorDataset(
        val_turkish_tokens["input_ids"], val_turkish_tokens["attention_mask"],
        val_english_tokens["input_ids"], val_english_tokens["attention_mask"]
    )

    test_dataset = torch.utils.data.TensorDataset(
        test_turkish_tokens["input_ids"], test_turkish_tokens["attention_mask"],
        test_english_tokens["input_ids"], test_english_tokens["attention_mask"]
    )
    return train_dataset, val_dataset, test_dataset

#preparing the dataset
data_path = "/home/aniyazi/school/CMP711/project/datasets/MaCoCu-tr-en.sent.txt/en-tr.deduped.txt"
data = []
with open(data_path, "r") as f:
    f.readline()
    for line in f:
        line_splitted = line.strip().split("\t")
        data.append([line_splitted[2], line_splitted[3]])


train_dataset, val_dataset, test_dataset = prepare_data(data)

# Example: Accessing a batch from the training dataset
example_batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=2)))
turkish_input_ids, turkish_attention_mask, english_input_ids, english_attention_mask = example_batch

# Print the tokenized tensors
print("Turkish Input IDs:", turkish_input_ids)
print("Turkish Attention Mask:", turkish_attention_mask)
print("English Input IDs:", english_input_ids)
print("English Attention Mask:", english_attention_mask)




vocab_size = 10000
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
max_len = 200
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


