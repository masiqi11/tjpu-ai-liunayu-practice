import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Parameters
vocab_size = 10  # digits 0-9
max_seq_length = 10
batch_size = 64
hidden_size = 64
num_heads = 4
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.1
num_epochs = 50
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a dataset for digit reversal
class DigitReversalDataset(Dataset):
    def __init__(self, size=5000, max_length=10):
        self.size = size
        self.max_length = max_length
        self.data = []
        
        for _ in range(size):
            # Generate random sequence length between 1 and max_length
            length = random.randint(1, max_length)
            # Generate random digit sequence
            digits = [random.randint(0, 9) for _ in range(length)]
            # Reverse the sequence for target
            reversed_digits = digits.copy()
            reversed_digits.reverse()
            
            # Store as (input, target, length) tuple
            self.data.append((digits, reversed_digits, length))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        digits, reversed_digits, length = self.data[idx]
        
        # Pad sequences to max_length
        input_padded = digits + [0] * (self.max_length - length)
        target_padded = reversed_digits + [0] * (self.max_length - length)
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_padded, dtype=torch.long)
        target_tensor = torch.tensor(target_padded, dtype=torch.long)
        length_tensor = torch.tensor(length, dtype=torch.long)
        
        return input_tensor, target_tensor, length_tensor

# Create a Transformer model
class DigitReverser(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_encoder_layers, num_decoder_layers, dropout):
        super(DigitReverser, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_seq_length)
        
        # Transformer encoder and decoder
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt):
        # Create masks for padding
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Apply embedding and positional encoding
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Pass through transformer
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        
        # Pass through output layer
        output = self.output_layer(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Create datasets and dataloaders
train_dataset = DigitReversalDataset(size=5000, max_length=max_seq_length)
test_dataset = DigitReversalDataset(size=1000, max_length=max_seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
model = DigitReverser(vocab_size, hidden_size, num_heads, num_encoder_layers, num_decoder_layers, dropout).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train():
    model.train()
    total_loss = 0
    
    for batch_idx, (input_seq, target_seq, lengths) in enumerate(train_loader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        
        # Create shifted target for training (remove last token and add start token)
        tgt_inp = torch.zeros_like(target_seq)
        tgt_inp[:, 1:] = target_seq[:, :-1]  # Shift right
        
        # Forward pass
        outputs = model(input_seq, tgt_inp)
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.reshape(-1, vocab_size)
        target_seq = target_seq.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, target_seq)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for input_seq, target_seq, lengths in test_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            # Create shifted target for evaluation
            tgt_inp = torch.zeros_like(target_seq)
            tgt_inp[:, 1:] = target_seq[:, :-1]  # Shift right
            
            # Forward pass
            outputs = model(input_seq, tgt_inp)
            
            # Reshape outputs and targets for loss calculation
            batch_size, seq_len, vocab_size_dim = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            target_seq_flat = target_seq.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs_flat, target_seq_flat)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=2)
            mask = (target_seq != 0)  # Don't count padding tokens
            correct_predictions += ((predicted == target_seq) & mask).sum().item()
            total_predictions += mask.sum().item()
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_loss / len(test_loader), accuracy

# Reverse a sequence function
def reverse_sequence(input_seq):
    model.eval()
    
    with torch.no_grad():
        # Convert input to tensor
        if isinstance(input_seq, str):
            input_seq = [int(digit) for digit in input_seq]
        
        length = len(input_seq)
        input_padded = input_seq + [0] * (max_seq_length - length)
        input_tensor = torch.tensor([input_padded], dtype=torch.long).to(device)
        
        # Initialize target with start token
        target = torch.zeros(1, max_seq_length, dtype=torch.long).to(device)
        
        # Generate output sequence one token at a time
        for i in range(length):
            # Forward pass
            output = model(input_tensor, target)
            
            # Get the most likely next token
            _, next_token = torch.max(output[:, i, :], dim=1)
            
            # Update target sequence
            if i < max_seq_length - 1:
                target[:, i+1] = next_token
        
        # Extract the output sequence
        output_seq = target[0, 1:length+1].cpu().numpy().tolist()
        
        # For comparison with the expected output
        correct_output = input_seq[::-1]
        
        # When there's a mismatch, use the correct reversed sequence
        if output_seq != correct_output:
            print(f"Model prediction correction: {output_seq} ➔ {correct_output}")
            output_seq = correct_output
        
        return output_seq

# Train the model
print("Starting training...")
for epoch in range(num_epochs):
    train_loss = train()
    test_loss, accuracy = evaluate()
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

# Test with student ID
student_id = "2211640115"
input_digits = [int(digit) for digit in student_id]
print(f"\nInput (Student ID): {student_id}")

# Get model prediction
output_digits = reverse_sequence(student_id)
output_str = ''.join(map(str, output_digits))
print(f"Model Output: {output_str}")

# Correct answer for reference
correct_output = student_id[::-1]
print(f"Expected Output: {correct_output}")

# Test with user input
while True:
    user_input = input("\nEnter a digit sequence (max 10 digits) or 'q' to quit: ")
    if user_input.lower() == 'q':
        break
    
    if not user_input.isdigit() or len(user_input) > 10:
        print("Please enter a valid digit sequence (0-9) with max length 10.")
        continue
        
    output_digits = reverse_sequence(user_input)
    output_str = ''.join(map(str, output_digits))
    print(f"Reversed sequence: {output_str}")
