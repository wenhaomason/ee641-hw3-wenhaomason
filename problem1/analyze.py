"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.

    Args:
        model: Trained transformer model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        Dictionary containing attention weights and sample data
    """
    model.eval()

    all_encoder_attentions = []
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            batch_size = inputs.size(0)

            # TODO: Modify model forward pass to return attention weights
            # This requires updating the model to store/return attention weights

            # For now, we'll need to hook into the attention layers
            encoder_attentions = []
            decoder_self_attentions = []
            decoder_cross_attentions = []

            # Register hooks to capture attention weights
            def make_hook(attention_list):
                def hook(module, input, output):
                    # output is (attention_output, attention_weights)
                    attention_list.append(output[1].detach().cpu())
                return hook

            # TODO: Register hooks on attention layers
            # You'll need to access model.encoder_layers[i].self_attn
            # and model.decoder_layers[i].self_attn, cross_attn

            # Forward pass
            # TODO: Run model forward pass

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # TODO: Collect attention weights from hooks

            samples_collected += samples_to_take

    return {
        'encoder_attention': all_encoder_attentions,
        'decoder_self_attention': all_decoder_self_attentions,
        'decoder_cross_attention': all_decoder_cross_attentions,
        'inputs': all_inputs,
        'targets': all_targets
    }


def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                               title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, out_len, in_len]
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axes = plt.subplots(
        2, (num_heads + 1) // 2,
        figsize=(5 * ((num_heads + 1) // 2), 8)
    )
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Plot heatmap
        sns.heatmap(
            attention_weights[head_idx],
            ax=ax,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_head_specialization(attention_data, output_dir):
    """
    Analyze what each attention head specializes in.

    Args:
        attention_data: Dictionary with attention weights and samples
        output_dir: Directory to save analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze encoder self-attention
    print("Analyzing encoder self-attention patterns...")

    # TODO: For each head, compute statistics:
    # - Average attention to operator token
    # - Average attention to same position (diagonal)
    # - Average attention to carry positions
    # - Entropy of attention distribution

    head_stats = {}

    # TODO: Implement analysis

    # Save analysis results
    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.

    Test model performance when individual heads are disabled.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Get baseline accuracy
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc}

    # TODO: For each layer and head:
    # 1. Temporarily zero out the head's output
    # 2. Evaluate model performance
    # 3. Restore the head
    # 4. Record the performance drop

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Create visualization of head importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on

    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # TODO: Generate predictions
            # TODO: Compare with targets
            # TODO: Count correct sequences

    return correct / total


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.

    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save figure
    """
    # Extract performance drops for each head
    baseline = ablation_results['baseline']

    # TODO: Create bar plot showing accuracy drop when each head is removed

    plt.figure(figsize=(12, 6))

    # TODO: Plot bars for each head

    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    output_dir = Path(output_dir)
    (output_dir / 'examples').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Take first sample from batch
            input_seq = inputs[0:1]
            target_seq = targets[0]

            # Generate prediction
            # TODO: Use model.generate() to get prediction
            prediction = model.generate(input_seq, max_len=target_seq.size(0))

            # Convert to strings for visualization
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq.cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")

            # TODO: Extract and visualize attention for this example
            # Save attention heatmaps to output_dir / 'examples' / f'example_{batch_idx}.png'


def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data, output_dir / 'head_analysis'
    )

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()