"""
Training script for sequence-to-sequence addition model.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from attention import create_causal_mask
from dataset import create_dataloaders, get_vocab_size
from model import Seq2SeqTransformer
from tqdm import tqdm


def compute_accuracy(outputs, targets, pad_token=0):
    """
    Compute sequence-level accuracy.

    Args:
        outputs: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth [batch, seq_len]
        pad_token: Padding token to ignore

    Returns:
        Accuracy (fraction of completely correct sequences)
    """
    # TODO: Get predicted tokens from logits
    preds = outputs.argmax(dim=-1)
    # TODO: Create mask for non-padding positions
    # For this task, targets are fixed-length (no padding), so use full mask
    non_pad_mask = torch.ones_like(targets, dtype=torch.bool)
    # TODO: Check if entire sequence matches (excluding padding)
    matches = (preds == targets) | (~non_pad_mask)
    seq_correct = matches.all(dim=1)
    return seq_correct.float().mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: Transformer model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Average loss, average accuracy
    """
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # TODO: Prepare decoder input and output for teacher forcing
        # Decoder input should be targets shifted right (exclude last token)
        # Decoder output should be targets shifted left (exclude first token)
        decoder_input = targets[:, :-1]
        decoder_output = targets[:, 1:]

        # TODO: Create causal mask for decoder (using shifted sequence length)
        # TODO: Forward pass
        # TODO: Compute loss
        # Hint: Flatten for cross entropy - need 2D tensors
        # TODO: Backward pass and optimization
        # TODO: Compute accuracy
        tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
        optimizer.zero_grad()
        outputs = model(inputs, decoder_input, tgt_mask=tgt_mask)
        vocab_size = outputs.size(-1)
        loss = criterion(outputs.reshape(-1, vocab_size), decoder_output.reshape(-1))
        loss.backward()
        optimizer.step()
        acc = compute_accuracy(outputs, decoder_output)

        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2%}'
        })

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Args:
        model: Transformer model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to run on

    Returns:
        Average loss, average accuracy
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # TODO: Prepare decoder input and output (same as training)
            # TODO: Create causal mask (using shifted sequence length)
            # TODO: Forward pass
            # TODO: Compute loss and accuracy (flatten for cross entropy)
            decoder_input = targets[:, :-1]
            decoder_output = targets[:, 1:]
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            outputs = model(inputs, decoder_input, tgt_mask=tgt_mask)
            vocab_size = outputs.size(-1)
            loss = criterion(outputs.reshape(-1, vocab_size), decoder_output.reshape(-1))

            # Compute accuracy under teacher forcing to match training setup
            acc = compute_accuracy(outputs, decoder_output)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train addition transformer')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size
    )

    # Create model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(args.device)

    # TODO: Initialize optimizer (Adam recommended)
    # TODO: Initialize learning rate scheduler (ReduceLROnPlateau recommended)
    # TODO: Initialize loss function (use nn.CrossEntropyLoss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = -1
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, args.device
        )

        # TODO: Step learning rate scheduler (pass val_loss)
        scheduler.step(val_loss)

        # Log results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2%}")

    # Test final model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

    # Save training history
    training_history['test_loss'] = [test_loss]
    training_history['test_acc'] = [test_acc]
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()