import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_model(model, optimizer, epoch, file_path):
    """
    Save the model and optimizer state dictionaries along with the epoch.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Model saved at epoch {epoch} to {file_path}")


def load_model(model, optimizer, file_path, device):
    """
    Load the model and optimizer state dictionaries from a checkpoint.

    Returns:
        epoch (int): The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.to(device)
    print(f"Model loaded from {file_path}, starting at epoch {epoch}")
    return epoch

def train(model, data_loader, optimizer, criterion, device, num_epochs=10, save_path="transformer_model.pth"):
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for src, tgt in tqdm(data_loader):
            src, tgt = src.to(device), tgt.to(device)
            if torch.cuda.is_available():
              src, tgt = src.cuda(), tgt.cuda()

            optimizer.zero_grad()
            # Use teacher forcing: feed target input except the last token.
            # output = model(src, tgt[:, :-1])
            output = model(src, tgt[:, :-1])

            output_flat = output.reshape(-1, output.size(-1))
            target_flat = tgt[:, 1:].reshape(-1)
            # Compute loss using outputs and target tokens shifted by one.
            # Check shapes for debugging

            loss = criterion(output_flat,target_flat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        save_model(model, optimizer, epoch, save_path)
def translate(model, src_sentence, src_tokenizer, tgt_tokenizer, device, max_len=50):
    """
    Given a source sentence, translate it using the model.

    Args:
        model: The trained TransformerModel.
        src_sentence (str): The input sentence in the source language.
        src_tokenizer: Tokenizer for the source language.
        tgt_tokenizer: Tokenizer for the target language.
        device: torch.device to run inference on.
        max_len: Maximum length of the target sequence.

    Returns:
        str: The translated sentence.
    """
    model.eval()
    # Tokenize the source sentence and add EOS token.
    src_tokens = src_tokenizer.encode(src_sentence) + [src_tokenizer.eos_token_id]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  # (1, src_seq_len)

    # Start with the beginning-of-sequence token.
    tgt_tokens = [tgt_tokenizer.bos_token_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        # Choose the token with highest probability from the last output step.
        next_token = output.argmax(-1)[:, -1].item()
        if next_token == tgt_tokenizer.eos_token_id:
            break
        tgt_tokens.append(next_token)

    translated_sentence = tgt_tokenizer.decode(tgt_tokens)
    return translated_sentence

def evaluate(model, data_loader, criterion, device, pad_token_id):
    """
    Evaluates the model on the provided data_loader and plots batch loss and accuracy.
    
    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module): Loss function (e.g., nn.CrossEntropyLoss).
        device (torch.device): Device to perform computation on.
        pad_token_id (int): The token ID used for padding, which should be ignored in accuracy.
    
    Returns:
        avg_loss (float): Average loss over the dataset.
        avg_accuracy (float): Overall token-level accuracy (excluding pad tokens).
    """
    model.eval()  # set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    # Lists to store batch-wise metrics for plotting
    batch_losses = []
    batch_accuracies = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # For teacher forcing: 
            #   * The decoder input is all tokens except the last one.
            #   * The target for loss computation is all tokens except the first one.
            decoder_input = tgt[:, :-1]   # shape: [batch, seq_len - 1]
            target_output = tgt[:, 1:]      # shape: [batch, seq_len - 1]
            
            # Forward pass: model output shape -> [batch, seq_len - 1, vocab_size]
            output = model(src, decoder_input)
            
            # Flatten the output and target for loss computation:
            output_flat = output.contiguous().view(-1, output.size(-1))
            target_flat = target_output.contiguous().view(-1)
            
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Compute predictions (taking the token with highest logit)
            preds = output.argmax(dim=-1)  # shape: [batch, seq_len - 1]
            
            # Create a mask to ignore padded tokens in the target.
            # Assume pad_token_id is set appropriately.
            mask = target_output != pad_token_id
            
            # Count correct predictions where the mask is True.
            correct = (preds == target_output) & mask
            correct_count = correct.sum().item()
            tokens_count = mask.sum().item()
            
            batch_accuracy = correct_count / tokens_count if tokens_count > 0 else 0
            batch_accuracies.append(batch_accuracy)
            
            total_correct += correct_count
            total_tokens += tokens_count

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    # Plot the batch losses and accuracies
    plt.figure(figsize=(12, 5))
    
    # Plot Loss per batch
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, marker='o', label="Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Batch Losses")
    plt.legend()
    
    # Plot Accuracy per batch
    plt.subplot(1, 2, 2)
    plt.plot(batch_accuracies, marker='o', color='green', label="Accuracy")
    plt.xlabel("Batch Number")
    plt.ylabel("Accuracy")
    plt.title("Batch Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return avg_loss, avg_accuracy