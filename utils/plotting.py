from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import integrate
import scienceplots
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import torch
import numpy as np
plt.style.use(['science'])

# font sizes

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'legend.fontsize': 16})

def plot_seqs(time_seqs, type_seqs, time_delta_seqs, sequence_length, calculated_deltas, decoder_output, num_samples=5, file_name=None):
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        time_seq = time_seqs[i].detach().cpu().numpy()
        type_seq = type_seqs[i].detach().cpu().numpy()
        time_delta_seq = time_delta_seqs[i].detach().cpu().numpy()
        seq_len = sequence_length[i].item()
        pred_time_delta_seq = calculated_deltas[i].detach().cpu().numpy() if isinstance(calculated_deltas, torch.Tensor) else calculated_deltas
        pred_mark_logits = decoder_output.mark_logits[i].detach().cpu().numpy()

        axs[i, 0].plot(time_seq[:seq_len], type_seq[:seq_len], 'o-', label='True')
        axs[i, 0].plot(time_seq[:seq_len], np.argmax(pred_mark_logits[:seq_len], axis=-1), 'o-', label='Predicted')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Event Type')
        axs[i, 0].set_title(f'Sample {i+1} - Event Sequence')
        axs[i, 0].legend()

        axs[i, 1].plot(time_seq[:seq_len], time_delta_seq[:seq_len], 'o-', label='True')
        axs[i, 1].plot(time_seq[:seq_len], pred_time_delta_seq[:seq_len], 'o-', label='Calculated')
        axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_ylabel('Time Delta')
        axs[i, 1].set_title(f'Sample {i+1} - Time Delta Sequence')
        axs[i, 1].legend()

    plt.tight_layout()
    plt.savefig(file_name) if file_name else plt.show()

def plot_seqs_both(
    time_seqs,
    type_seqs,
    time_delta_seqs,
    sequence_length,
    tpp_eval_metrics,
    vae_eval_metrics,
    decoder_output_tpp,
    decoder_output_vae,
    num_samples=5,
    file_name=None,
):
    """
    Plot event sequences and compare deterministic vs probabilistic intensity functions.

    Args:
        time_seqs (torch.Tensor): Ground truth event times.
        type_seqs (torch.Tensor): Ground truth event types.
        time_delta_seqs (torch.Tensor): Ground truth time deltas.
        sequence_length (torch.Tensor): Sequence lengths.
        tpp_eval_metrics (dict): Evaluation metrics from the TPP model.
        vae_eval_metrics (dict): Evaluation metrics from the VAE model.
        decoder_output_tpp (object): Decoder output from TPP model.
        decoder_output_vae (object): Decoder output from VAE model.
        num_samples (int): Number of sequences to plot.
    """
    predicted_intervals_tpp = tpp_eval_metrics["predicted_intervals"]
    predicted_intervals_vae = vae_eval_metrics["predicted_intervals"]

    fig, axes = plt.subplots(num_samples, 2, figsize=(18, 5 * num_samples))

    for i in range(num_samples):
        time_seq = time_seqs[i].detach().cpu().numpy()
        type_seq = type_seqs[i].detach().cpu().numpy()
        time_delta_seq = time_delta_seqs[i].detach().cpu().numpy()
        seq_len = sequence_length[i].item()

        pred_time_delta_tpp = (
            predicted_intervals_tpp[i].detach().cpu().numpy()
            if isinstance(predicted_intervals_tpp, torch.Tensor)
            else predicted_intervals_tpp
        )
        pred_time_delta_vae = (
            predicted_intervals_vae[i].detach().cpu().numpy()
            if isinstance(predicted_intervals_vae, torch.Tensor)
            else predicted_intervals_vae
        )

        pred_event_types_tpp = torch.argmax(decoder_output_tpp.mark_logits[i], dim=-1).detach().cpu().numpy()
        pred_event_types_vae = torch.argmax(decoder_output_vae.mark_logits[i], dim=-1).detach().cpu().numpy()

        axes[i, 0].plot(time_seq[:seq_len], type_seq[:seq_len], "o-", label="True")
        axes[i, 0].plot(
            time_seq[:seq_len],
            pred_event_types_tpp[:seq_len],
            "o-",
            label="TPP Predicted",
        )
        axes[i, 0].plot(
            time_seq[:seq_len],
            pred_event_types_vae[:seq_len],
            "o-",
            label="VAE Predicted",
        )
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Event Type")
        axes[i, 0].set_title(f"Event Sequence {i+1}")
        axes[i, 0].legend()

        axes[i, 1].plot(time_seq[:seq_len], time_delta_seq[:seq_len], "o-", label="True")
        axes[i, 1].plot(
            time_seq[:seq_len],
            pred_time_delta_tpp[:seq_len],
            "o-",
            label="TPP Predicted",
        )
        axes[i, 1].plot(
            time_seq[:seq_len],
            pred_time_delta_vae[:seq_len],
            "o-",
            label="VAE Predicted",
        )
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Time Delta")
        axes[i, 1].set_title(f"Time Delta Sequence {i+1}")
        axes[i, 1].legend()

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()

def plot_latent_space_vae(model, data_loader, device, file_name=None):
    """
    Plot latent space of the VAE model using mu from the forward pass, excluding padding tokens.

    Args:
        model (VAETPPModel): VAE model.
        data_loader (DataLoader): Data loader.
        device (str): Device to run the model on.
    """
    model.eval()
    mus = []
    event_types = []
    masks = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            mu = output.mu
            
            # Create mask for padding tokens
            sequence_lengths = batch["sequence_length"]
            batch_mask = torch.arange(batch["type_seqs"].size(1)).unsqueeze(0).to(device) < sequence_lengths.unsqueeze(1)
            
            # Reshape mu and event types if they have sequence dimension
            if len(mu.shape) > 2:
                mu = mu.reshape(-1, mu.shape[-1])
                batch_mask = batch_mask.reshape(-1)
            
            mus.append(mu)
            event_types.append(batch["type_seqs"].reshape(-1))
            masks.append(batch_mask)
            
            print("Shape of mu:", mu.shape)
            print("Shape of mask:", batch_mask.shape)
    
    # Concatenate all batches
    mus = torch.cat(mus, dim=0).cpu().numpy()
    event_types = torch.cat(event_types, dim=0).cpu().numpy()
    masks = torch.cat(masks, dim=0).cpu().numpy()
    
    print("Shape before masking:")
    print("mus:", mus.shape)
    print("event_types:", event_types.shape)
    print("masks:", masks.shape)
    
    # Apply mask to filter out padding tokens
    mus = mus[masks]
    event_types = event_types[masks]
    
    print("\nShape after masking:")
    print("mus:", mus.shape)
    print("event_types:", event_types.shape)
    
    # Reduce dimensionality using t-SNE    
    tsne = TSNE(n_components=2, random_state=0)
    mus = tsne.fit_transform(mus)
    
    # Make color discrete map using default rc colormap
    unique_event_types = np.unique(event_types)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = plt.cm.tab20.colors
    color_map = {event_type: colors[i % len(colors)] for i, event_type in enumerate(unique_event_types)}
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Create list of colors for each point
    point_colors = [color_map[int(et)] for et in event_types]
    
    scatter = plt.scatter(mus[:, 0], mus[:, 1], c=point_colors, alpha=0.6)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space of VAETPP Model (Excluding Padding)")
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[et], label=f'Event {et}',
                                markersize=10)
                      for et in unique_event_types]
    plt.legend(handles=legend_elements, title="Event Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(file_name) if file_name else plt.show()
    
    
def plot_latent_space_tpp(model, data_loader, device, file_name=None):
    """
    Plot latent space for TPP models using hidden states, excluding padding tokens.

    Args:
        model (nn.Module): TPP model (e.g., RMTPPModel).
        data_loader (torch.utils.data.DataLoader): Data loader for input batches.
        device (torch.device): Device to run the model on.
    """
    model.eval()
    latent_representations = []
    event_types = []
    masks = []

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass to get hidden states
            hidden_states = model.encoder(batch)  # Extract hidden states from encoder
            
            # Create mask to exclude padding tokens
            sequence_lengths = batch["sequence_length"]
            batch_mask = (
                torch.arange(batch["type_seqs"].size(1)).unsqueeze(0).to(device)
                < sequence_lengths.unsqueeze(1)
            )
            
            # Reshape hidden states and masks if they have a sequence dimension
            if len(hidden_states.shape) > 2:  # If hidden states are [batch_size, seq_len, hidden_dim]
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                batch_mask = batch_mask.reshape(-1)
            
            latent_representations.append(hidden_states)
            event_types.append(batch["type_seqs"].reshape(-1))
            masks.append(batch_mask)
    
    # Concatenate all batches
    latent_representations = torch.cat(latent_representations, dim=0).cpu().numpy()
    event_types = torch.cat(event_types, dim=0).cpu().numpy()
    masks = torch.cat(masks, dim=0).cpu().numpy()
    
    # Apply mask to exclude padding tokens
    latent_representations = latent_representations[masks]
    event_types = event_types[masks]
    
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    latent_representations = tsne.fit_transform(latent_representations)
    
    # Create discrete color mapping for event types
    unique_event_types = np.unique(event_types)
    colors = plt.cm.tab20.colors  # Use tab20 colormap for discrete categories
    color_map = {event_type: colors[i % len(colors)] for i, event_type in enumerate(unique_event_types)}
    
    # Create a scatter plot of the latent space
    plt.figure(figsize=(10, 10))
    point_colors = [color_map[int(et)] for et in event_types]
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=point_colors, alpha=0.6)
    
    # Add axis labels and title
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space of RMTPP Model")
    
    # Add legend for event types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[et],
                   label=f'Event {et}', markersize=10)
        for et in unique_event_types
    ]
    plt.legend(handles=legend_elements, title="Event Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()
    
    
def plot_samples_vae(model, data_loader, device, max_samples=100):
    model.eval()
    
    batch = next(iter(data_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    sequence_lengths = batch["sequence_length"].to(device)
    
    # identify the padding tokens
    mask = torch.arange(batch["type_seqs"].size(1)).unsqueeze(0).to(device) < sequence_lengths.unsqueeze(1)
    mask = mask.cpu().numpy()
    
    with torch.no_grad():
        output = model(batch)
        mu = output.mu.cpu().numpy()
        logvar = output.logvar.cpu().numpy()
        event_types = batch["type_seqs"].cpu().numpy()
        
        mu = mu[mask].reshape(-1, mu.shape[-1])
        logvar = logvar[mask].reshape(-1, logvar.shape[-1])
        event_types = event_types[mask].reshape(-1)
        
        if len(mu) > max_samples:
            idx = np.random.choice(len(mu), max_samples, replace=False)
            mu = mu[idx]
            logvar = logvar[idx]
            event_types = event_types[idx]
    
    std = np.exp(0.5 * logvar)
    n_samples_per_point = 5
    samples = []
    sample_types = []
    
    for i in range(len(mu)):
        for _ in range(n_samples_per_point):
            sample = mu[i] + np.random.randn(*mu[i].shape) * std[i]
            samples.append(sample)
            sample_types.append(event_types[i])
    
    samples = np.stack(samples)
    sample_types = np.array(sample_types)
    
    tsne = TSNE(n_components=2, random_state=42)
    samples_2d = tsne.fit_transform(samples)
    
    plt.figure(figsize=(12, 8))
    
    # Get default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, event_type in enumerate(np.unique(event_types)):
        mask = sample_types == event_type
        points = samples_2d[mask]
        
        kde = gaussian_kde(points.T)
        x_min, x_max = points[:, 0].min() - 0.5, points[:, 0].max() + 0.5
        y_min, y_max = points[:, 1].min() - 0.5, points[:, 1].max() + 0.5
        
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        color = colors[idx % len(colors)]
        label = f'event {event_type + 1}'
        
        plt.contour(X, Y, Z, levels=5, colors=color, alpha=0.3)
        plt.scatter(points[:, 0], points[:, 1], c=color, alpha=0.4, label=label, s=30)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.title('Samples from VAETPP latent space')
    plt.savefig('vae_latent_space_samples.png')
    plt.show()
    return plt.gcf()