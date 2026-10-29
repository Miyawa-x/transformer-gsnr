"""Loss components used by the original fixed-momentum DeepKD reproduction."""

import torch
import torch.nn.functional as F


def _ground_truth_mask(logits, target):
    target = target.reshape(-1)
    return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()


def _other_class_mask(logits, target):
    target = target.reshape(-1)
    return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()


def _binary_probabilities(probabilities, ground_truth_mask, other_class_mask):
    target_probability = (probabilities * ground_truth_mask).sum(
        dim=1, keepdim=True
    )
    other_probability = (probabilities * other_class_mask).sum(
        dim=1, keepdim=True
    )
    return torch.cat([target_probability, other_probability], dim=1)


def dkd_component_losses(
    logits_student,
    logits_teacher,
    target,
    temperature: float,
):
    """Return unweighted TCKD and NCKD losses matching official MDistiller."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    ground_truth_mask = _ground_truth_mask(logits_student, target)
    other_class_mask = _other_class_mask(logits_student, target)

    student_probabilities = F.softmax(logits_student / temperature, dim=1)
    teacher_probabilities = F.softmax(logits_teacher / temperature, dim=1)
    student_binary = _binary_probabilities(
        student_probabilities, ground_truth_mask, other_class_mask
    )
    teacher_binary = _binary_probabilities(
        teacher_probabilities, ground_truth_mask, other_class_mask
    )
    tckd_loss = (
        F.kl_div(torch.log(student_binary), teacher_binary, reduction="sum")
        * temperature**2
        / target.shape[0]
    )

    teacher_non_target = F.softmax(
        logits_teacher / temperature - 1000.0 * ground_truth_mask,
        dim=1,
    )
    student_non_target_log = F.log_softmax(
        logits_student / temperature - 1000.0 * ground_truth_mask,
        dim=1,
    )
    nckd_loss = (
        F.kl_div(student_non_target_log, teacher_non_target, reduction="sum")
        * temperature**2
        / target.shape[0]
    )
    return tckd_loss, nckd_loss

