import torch
from tqdm import tqdm


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
