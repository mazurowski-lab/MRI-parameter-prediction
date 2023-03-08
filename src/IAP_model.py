import torch
import torch.nn as nn
import numpy as np

# IAP recognition loss
criterion_CE = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()
verbose_loss = False
weight_classification = 1
weight_regression = 1

def get_total_criterion(all_feature_names, device):
    def total_criterion(outputs, targets, get_topk_counts=False, max_k=2, relative_error=False, return_predictions=False):
        #total loss for all features
        total_loss = 0
        feature_losses = {}
        feature_correct_classification_counts = {}
        predictions = {}
        if get_topk_counts:
            topk_feature_correct_classification_counts = {}

        for feature_idx, feature_name in enumerate(targets.keys()):
            start_output_idx = int(np.array(list(all_feature_names.values())[:feature_idx]).sum())
            if verbose_loss:
                print(feature_idx, feature_name, start_output_idx)

            num_feature_options = all_feature_names[feature_name]
            outputs_this_feature = outputs[:, start_output_idx: start_output_idx + num_feature_options]
            targets_this_feature = targets[feature_name].to(device).float()
            # regression
            if all_feature_names[feature_name] == 1:
                outputs_this_feature = outputs_this_feature.flatten()
                criterion = criterion_MSE
                weight = weight_regression

                if return_predictions:
                    predictions[feature_name] = outputs_this_feature
            # classification
            else:
                targets_this_feature = targets_this_feature.long()
                criterion = criterion_CE
                weight = weight_classification

                # determine if classifications are correct for accuracy measurement
                # Calculate predicted labels
                _, predicted = torch.max(outputs_this_feature.data, 1)
                correct_examples = (predicted == targets_this_feature).sum().item()
                feature_correct_classification_counts[feature_name] = correct_examples

                if return_predictions:
                    predictions[feature_name] = predicted

                # get top-k correct counts
                if get_topk_counts:
                    for k in range(2, max_k+1): # k=1 already covered
                        if k not in topk_feature_correct_classification_counts.keys():
                            topk_feature_correct_classification_counts[k] = {}

                        _, topk_predicted = torch.topk(outputs_this_feature.data, k, dim=1)
                        topk_correct_examples = (topk_predicted == targets_this_feature.unsqueeze(1)).sum().item()
                        topk_feature_correct_classification_counts[k][feature_name] = topk_correct_examples


            feature_loss = criterion(outputs_this_feature, targets_this_feature)
            feature_losses[feature_name] = feature_loss.item()
            total_loss += weight * feature_loss
            if verbose_loss:
                print(feature_name, loss)

        ret = [total_loss, feature_losses, feature_correct_classification_counts]
        if get_topk_counts:
            ret.append(topk_feature_correct_classification_counts)
        if return_predictions:
            ret.append(predictions)
        return ret

    return total_criterion