"""
Analysis and visualization of attention patterns.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from attention import create_causal_mask
from dataset import create_dataloaders, get_vocab_size
from model import Seq2SeqTransformer
from tqdm import tqdm


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
        progress = tqdm(total=num_samples, desc="Extracting attentions")
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
            handles = []
            for layer in model.encoder_layers:
                handles.append(layer.self_attn.register_forward_hook(make_hook(encoder_attentions)))
            for layer in model.decoder_layers:
                handles.append(layer.self_attn.register_forward_hook(make_hook(decoder_self_attentions)))
                handles.append(layer.cross_attn.register_forward_hook(make_hook(decoder_cross_attentions)))

            # Forward pass
            # TODO: Run model forward pass
            # Use teacher forcing style inputs for decoder to align positions
            decoder_input = targets[:, :-1]
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            _ = model(inputs, decoder_input, tgt_mask=tgt_mask)

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # TODO: Collect attention weights from hooks
            # Slice captured attentions to match collected samples
            # Encoder self-attn
            for t in encoder_attentions:
                all_encoder_attentions.append(t[:samples_to_take])
            # Decoder self-attn
            for t in decoder_self_attentions:
                all_decoder_self_attentions.append(t[:samples_to_take])
            # Decoder cross-attn
            for t in decoder_cross_attentions:
                all_decoder_cross_attentions.append(t[:samples_to_take])

            # Remove hooks
            for h in handles:
                h.remove()

            samples_collected += samples_to_take
            progress.update(samples_to_take)

        progress.close()

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
    plt.close(fig)


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
    # We aggregate across collected encoder attention tensors
    # Each tensor has shape [batch, num_heads, seq_len, seq_len]
    if attention_data['encoder_attention']:
        # Concatenate along batch dimension from all captures
        enc_attn_list = attention_data['encoder_attention']
        enc_all = torch.cat(enc_attn_list, dim=0)  # [N, H, L, L]
        num_heads = enc_all.size(1)
        seq_len = enc_all.size(2)

        # Operator token id is 10 per dataset generation
        operator_token_id = 10

        # Build mask for operator positions per sample using inputs provided
        # inputs are saved as numpy arrays with padding possibly 0; operator token appears once in input
        if isinstance(attention_data['inputs'], list):
            # List of 1D arrays -> stack to [N, L]
            inputs_all = np.stack(attention_data['inputs'], axis=0)
        else:
            inputs_all = np.array(attention_data['inputs'])
        # Ensure shape [N, L]
        if inputs_all.ndim != 2:
            inputs_all = inputs_all.reshape(-1, enc_all.size(-1))
        N = enc_all.size(0)
        # Clip if there is any mismatch
        min_N = min(N, inputs_all.shape[0])
        enc_all = enc_all[:min_N]
        inputs_all = inputs_all[:min_N]

        # Operator mask [N, 1, 1, L]
        op_mask = (torch.from_numpy(inputs_all) == operator_token_id).unsqueeze(1).unsqueeze(1)
        op_mask = op_mask.to(enc_all.dtype)

        # Average attention to operator token (average over queries and samples)
        # Compute mean over query positions and samples of attention assigned to positions where op_mask==1
        op_attn = (enc_all * op_mask).sum(dim=-1) / (op_mask.sum(dim=-1).clamp(min=1.0))
        avg_op_attn = op_attn.mean(dim=(0, 2)).tolist()  # per-head

        # Diagonal attention
        diag_indices = torch.arange(seq_len)
        diag_attn = enc_all[:, :, diag_indices, diag_indices]  # [N, H, L]
        avg_diag_attn = diag_attn.mean(dim=(0, 2)).tolist()

        # Carry positions (approximate as attention to previous position i-1)
        if seq_len > 1:
            prev_indices = torch.clamp(diag_indices - 1, min=0)
            carry_attn = enc_all[:, :, diag_indices, prev_indices]  # [N, H, L]
            avg_carry_attn = carry_attn.mean(dim=(0, 2)).tolist()
        else:
            avg_carry_attn = [0.0 for _ in range(num_heads)]

        # Entropy of attention distribution over keys per head/query
        # entropy = -sum p log p; add small epsilon for stability
        eps = 1e-9
        p = enc_all.clamp(min=eps)
        entropy = -(p * p.log()).sum(dim=-1)  # [N, H, L]
        avg_entropy = entropy.mean(dim=(0, 2)).tolist()

        for h in range(num_heads):
            head_stats[f'head_{h}'] = {
                'avg_attention_to_operator': float(avg_op_attn[h]),
                'avg_attention_diagonal': float(avg_diag_attn[h]),
                'avg_attention_carry_prev': float(avg_carry_attn[h]),
                'avg_entropy': float(avg_entropy[h])
            }

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
    def eval_model():
        return evaluate_model(model, dataloader, device)

    # Helper to ablate one head by zeroing corresponding columns of linear_out
    def ablate_head_linear_out(mha_module, head_idx):
        d_k = mha_module.d_k
        start = head_idx * d_k
        end = (head_idx + 1) * d_k
        W = mha_module.linear_out.weight
        # Save original slice
        original = W[:, start:end].detach().clone()
        # Zero out
        with torch.no_grad():
            W[:, start:end] = 0
        return original

    def restore_head_linear_out(mha_module, head_idx, original_slice):
        d_k = mha_module.d_k
        start = head_idx * d_k
        end = (head_idx + 1) * d_k
        with torch.no_grad():
            mha_module.linear_out.weight[:, start:end] = original_slice

    # Encoder self-attention ablation
    ablation_results['encoder'] = {}
    for layer_idx, layer in enumerate(model.encoder_layers):
        layer_results = []
        for head_idx in range(layer.self_attn.num_heads):
            orig = ablate_head_linear_out(layer.self_attn, head_idx)
            acc = eval_model()
            restore_head_linear_out(layer.self_attn, head_idx, orig)
            layer_results.append(acc)
        ablation_results['encoder'][str(layer_idx)] = layer_results

    # Decoder self-attention ablation
    ablation_results['decoder_self'] = {}
    for layer_idx, layer in enumerate(model.decoder_layers):
        layer_results = []
        for head_idx in range(layer.self_attn.num_heads):
            orig = ablate_head_linear_out(layer.self_attn, head_idx)
            acc = eval_model()
            restore_head_linear_out(layer.self_attn, head_idx, orig)
            layer_results.append(acc)
        ablation_results['decoder_self'][str(layer_idx)] = layer_results

    # Decoder cross-attention ablation
    ablation_results['decoder_cross'] = {}
    for layer_idx, layer in enumerate(model.decoder_layers):
        layer_results = []
        for head_idx in range(layer.cross_attn.num_heads):
            orig = ablate_head_linear_out(layer.cross_attn, head_idx)
            acc = eval_model()
            restore_head_linear_out(layer.cross_attn, head_idx, orig)
            layer_results.append(acc)
        ablation_results['decoder_cross'][str(layer_idx)] = layer_results

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
            preds = model.generate(inputs, max_len=targets.size(1))
            matches = (preds == targets)
            seq_correct = matches.all(dim=1)
            correct += seq_correct.sum().item()
            total += targets.size(0)

    return (correct / total) if total > 0 else 0.0


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

    fig = plt.figure(figsize=(12, 6))

    # TODO: Plot bars for each head
    labels = []
    drops = []
    if 'encoder' in ablation_results:
        for layer_idx, acc_list in ablation_results['encoder'].items():
            for h, acc in enumerate(acc_list):
                labels.append(f'Encoder L{layer_idx} H{h}')
                drops.append(baseline - acc)
    if 'decoder_self' in ablation_results:
        for layer_idx, acc_list in ablation_results['decoder_self'].items():
            for h, acc in enumerate(acc_list):
                labels.append(f'Decoder Self L{layer_idx} H{h}')
                drops.append(baseline - acc)
    if 'decoder_cross' in ablation_results:
        for layer_idx, acc_list in ablation_results['decoder_cross'].items():
            for h, acc in enumerate(acc_list):
                labels.append(f'Decoder Cross L{layer_idx} H{h}')
                drops.append(baseline - acc)

    x = np.arange(len(labels))
    plt.bar(x, drops)
    plt.xticks(x, labels, rotation=45, ha='right')

    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)

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
            # Capture decoder cross-attention for first decoder layer
            dec_cross_attns = []
            def hook(module, inp, out):
                dec_cross_attns.append(out[1].detach().cpu())
            handle = model.decoder_layers[0].cross_attn.register_forward_hook(hook)

            # Run one teacher-forced forward to populate attention
            decoder_input = target_seq.unsqueeze(0)[:, :-1]
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            _ = model(input_seq, decoder_input, tgt_mask=tgt_mask)
            handle.remove()

            if dec_cross_attns:
                # Take first capture, first sample
                attn = dec_cross_attns[0][0]  # [num_heads, out_len, in_len]
                # Build token labels
                in_tokens = [str(int(t)) for t in input_seq[0].cpu().numpy().tolist()]
                out_tokens = [str(int(t)) for t in target_seq.cpu().numpy().tolist()[:-1]]  # aligned with decoder_input
                save_path = output_dir / 'attention_patterns' / f'example_{batch_idx}.png'
                visualize_attention_pattern(attn, in_tokens, out_tokens,
                                            title=f'Example {batch_idx+1} Decoder Cross-Attention',
                                            save_path=save_path)


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
    print(f"Computed head specialization stats for {len(head_stats)} heads (if any).")

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )
    print(f"Ablation study completed. Entries recorded: {len(ablation_results)}")

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()