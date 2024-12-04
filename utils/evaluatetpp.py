import numpy as np
from scipy import integrate
import torch
from torch.utils.data import DataLoader


def evaluate_tpp_model(model, test_dataloader, config, device):
    """
    Evaluate TPP model performance on test data.

    Args:
        model: TPP model instance
        test_dataloader: DataLoader for test data
        config: Configuration object containing model parameters
        device: torch device

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        event_total = 0
        all_cnt = np.zeros(config.num_event_types)
        acc_cnt = np.zeros(config.num_event_types)
        pre_cnt = np.zeros(config.num_event_types)
        time_error = 0
        time_mse = 0
        predicted_intervals = []

        for batch in test_dataloader:
            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            # Get model predictions
            decoder_output = model(batch)

            # Get batch tensors
            time_seqs = batch["time_seqs"].cpu().numpy()  # absolute times
            time_delta_seqs = batch["time_delta_seqs"].cpu().numpy()  # time intervals
            sequence_length = batch["sequence_length"]

            # Calculate event type accuracy
            event_total += sequence_length.sum().item()

            # Get predicted event types
            mark_logits = decoder_output.mark_logits
            event_output = torch.argmax(mark_logits, dim=-1).cpu().numpy()
            event_target = batch["type_seqs"].cpu().numpy()

            # Create mask for valid (non-padding) positions
            mask = torch.arange(event_target.shape[1])[None, :].to(
                device
            ) < sequence_length[:, None].to(device)
            mask = mask.cpu().numpy()

            # Calculate event type statistics using utility function
            all_cnt_batch, acc_cnt_batch, pre_cnt_batch = calculate_event_metrics(
                event_output,
                event_target,
                sequence_length.cpu().numpy(),
                config.num_event_types,
            )
            all_cnt += all_cnt_batch
            acc_cnt += acc_cnt_batch
            pre_cnt += pre_cnt_batch

            # Calculate time prediction error
            time_output = (
                decoder_output.time_output.squeeze(-1).cpu().numpy()
            )  # [batch_size, seq_len]
            intensity_w = model.decoder.intensity_w.cpu().data.numpy()
            intensity_b = model.decoder.intensity_b.cpu().data.numpy()

            # Calculate expected time for each position in each sequence
            for i in range(time_output.shape[0]):  # For each sequence
                for j in range(time_output.shape[1]):  # For each position
                    if mask[i, j]:  # Only consider non-padding positions
                        # Get last time (time of previous event)
                        last_time = time_seqs[i, j - 1] if j > 0 else 0.0

                        # Get current history event value
                        history_event = time_output[i, j]

                        # Integrate to get expected absolute time of next event
                        expected_next_time = integrate.quad(
                            lambda t: (t + last_time)
                            * np.exp(
                                history_event
                                + intensity_w * t
                                + intensity_b
                                + (
                                    np.exp(history_event + intensity_b)
                                    - np.exp(
                                        history_event + intensity_w * t + intensity_b
                                    )
                                )
                                / intensity_w
                            ),
                            0,
                            np.inf,
                        )[0]

                        # Calculate time delta error
                        predicted_interval = expected_next_time - last_time
                        actual_interval = time_delta_seqs[i, j]
                        time_error += np.abs(predicted_interval - actual_interval)
                        time_mse += (predicted_interval - actual_interval) ** 2
                        predicted_intervals.append(predicted_interval)

        # Calculate final metrics
        time_mae = time_error / event_total
        time_rmse = np.sqrt(time_mse / event_total)
        event_accuracy = acc_cnt.sum() / all_cnt.sum()

        # Calculate per-type metrics
        type_metrics = {}
        for event_type in range(config.num_event_types):
            if all_cnt[event_type] > 0:
                precision = (
                    acc_cnt[event_type] / pre_cnt[event_type]
                    if pre_cnt[event_type] > 0
                    else 0
                )
                recall = acc_cnt[event_type] / all_cnt[event_type]
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                type_metrics[f"type_{event_type}"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

    return {
        "time_mae": time_mae,
        "time_rmse": time_rmse,
        "event_accuracy": event_accuracy,
        "event_total": event_total,
        "type_metrics": type_metrics,
        "predicted_intervals": predicted_intervals,
    }


def evaluate_vae_model(model, test_dataloader, config, device):
    model.eval()

    with torch.no_grad():
        event_total = 0
        all_cnt = np.zeros(config.num_event_types)
        acc_cnt = np.zeros(config.num_event_types)
        pre_cnt = np.zeros(config.num_event_types)
        time_error = 0
        time_mse = 0
        predicted_intervals = []

        for batch in test_dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Get model predictions
            decoder_output = model(batch)
            if isinstance(decoder_output, tuple):
                time_output, mark_logits, *latent_outputs = decoder_output
                latent_means = latent_outputs[0] if len(latent_outputs) > 0 else None
                latent_logvars = latent_outputs[1] if len(latent_outputs) > 1 else None
            elif isinstance(decoder_output, dict):
                time_output = decoder_output["time_output"]
                mark_logits = decoder_output["mark_logits"]
                latent_means = decoder_output.get("latent_means", None)
                latent_logvars = decoder_output.get("latent_logvars", None)
            else:
                raise TypeError("Unexpected decoder output type")

            time_output = time_output.squeeze(-1).cpu().numpy() if time_output is not None else None
            latent_means = latent_means.cpu().numpy() if latent_means is not None else None
            latent_logvars = latent_logvars.cpu().numpy() if latent_logvars is not None else None

            time_seqs = batch["time_seqs"].cpu().numpy()
            time_delta_seqs = batch["time_delta_seqs"].cpu().numpy()
            sequence_length = batch["sequence_length"]

            event_total += sequence_length.sum().item()

            event_output = torch.argmax(mark_logits, dim=-1).cpu().numpy()
            event_target = batch["type_seqs"].cpu().numpy()

            mask = torch.arange(event_target.shape[1])[None, :].to(device) < sequence_length[:, None].to(device)
            mask = mask.cpu().numpy()

            all_cnt_batch, acc_cnt_batch, pre_cnt_batch = calculate_event_metrics(
                event_output, event_target, sequence_length.cpu().numpy(), config.num_event_types
            )
            all_cnt += all_cnt_batch
            acc_cnt += acc_cnt_batch
            pre_cnt += pre_cnt_batch

            for i in range(time_output.shape[0]):
                for j in range(time_output.shape[1]):
                    if mask[i, j]:
                        last_time = time_seqs[i, j - 1] if j > 0 else 0.0
                        predicted_interval = time_output[i, j]
                        actual_interval = time_delta_seqs[i, j]
                        time_error += np.abs(predicted_interval - actual_interval)
                        time_mse += (predicted_interval - actual_interval) ** 2
                        predicted_intervals.append(predicted_interval)

        time_mae = time_error / event_total
        time_rmse = np.sqrt(time_mse / event_total)
        event_accuracy = acc_cnt.sum() / all_cnt.sum()

        type_metrics = {}
        for event_type in range(config.num_event_types):
            if all_cnt[event_type] > 0:
                precision = acc_cnt[event_type] / pre_cnt[event_type] if pre_cnt[event_type] > 0 else 0
                recall = acc_cnt[event_type] / all_cnt[event_type]
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                type_metrics[f"type_{event_type}"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

    return {
        "time_mae": time_mae,
        "time_rmse": time_rmse,
        "event_accuracy": event_accuracy,
        "event_total": event_total,
        "type_metrics": type_metrics,
        "predicted_intervals": predicted_intervals,
        "latent_means": latent_means,
        "latent_logvars": latent_logvars,
    }


def calculate_event_metrics(
    event_output, event_target, sequence_length, num_event_types
):
    """
    Calculate event type statistics.

    Args:
        event_output (np.ndarray): Predicted event types.
        event_target (np.ndarray): True event types.
        sequence_length (np.ndarray): Lengths of each sequence.
        num_event_types (int): Total number of event types.

    Returns:
        tuple: (all_cnt, acc_cnt, pre_cnt) the number of true events,
        the number of correctly predicted events,
        and the number of predicted events for each event type.
    """
    all_cnt = np.zeros(num_event_types)
    acc_cnt = np.zeros(num_event_types)
    pre_cnt = np.zeros(num_event_types)

    mask = np.arange(event_target.shape[1])[None, :] < sequence_length[:, None]

    for i in range(event_target.shape[0]):
        for j in range(event_target.shape[1]):
            if mask[i, j]:
                all_cnt[event_target[i, j]] += 1
                pre_cnt[event_output[i, j]] += 1
                if event_output[i, j] == event_target[i, j]:
                    acc_cnt[event_output[i, j]] += 1
    return all_cnt, acc_cnt, pre_cnt
