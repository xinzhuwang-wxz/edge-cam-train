"""命门尺子 · 层级可用率（docs/classify/06 §A）。先红后绿。"""

import torch

from edge_cam.eval.hierarchical import Hierarchy, hierarchical_usability


def _hier() -> Hierarchy:
    # 5 类；genus A:{0,1} B:{2,3} C:{4}；family X:{0,1,2} Y:{3,4}
    return Hierarchy(genus=["A", "A", "B", "B", "C"], family=["X", "X", "X", "Y", "Y"])


def _from_probs(probs: list[list[float]]) -> torch.Tensor:
    """构造 logits 使 softmax(logits) 精确等于给定概率。"""
    return torch.log(torch.tensor(probs))


def test_confident_correct_species_is_usable():
    m = hierarchical_usability(
        _from_probs([[0.9, 0.05, 0.02, 0.02, 0.01]]), torch.tensor([0]), _hier()
    )
    assert m.usable_rate == 1.0
    assert m.species_correct == 1 and m.critical_error == 0


def test_confident_wrong_species_is_critical_error():
    # 种级高置信押 class0，但真值是同属的 class1 → 自信报错种
    m = hierarchical_usability(
        _from_probs([[0.9, 0.05, 0.02, 0.02, 0.01]]), torch.tensor([1]), _hier()
    )
    assert m.usable_rate == 0.0
    assert m.critical_error == 1 and m.species_report == 1


def test_fallback_to_correct_genus_is_usable():
    # 种级最高 0.35 < 0.5 不过门；属 A = 0.30+0.35 = 0.65 过门，真属 A → 可用（报属）
    m = hierarchical_usability(
        _from_probs([[0.30, 0.35, 0.20, 0.10, 0.05]]), torch.tensor([0]), _hier()
    )
    assert m.usable_rate == 1.0
    assert m.report_genus == 1 and m.species_report == 0


def test_fallback_genus_wrong_not_usable():
    # 属 A=0.65 过门但真值是 class4（属 C）→ 报属但属错，不可用
    m = hierarchical_usability(
        _from_probs([[0.30, 0.35, 0.20, 0.10, 0.05]]), torch.tensor([4]), _hier()
    )
    assert m.usable_rate == 0.0
    assert m.report_genus == 1


def test_fallback_to_family_is_usable():
    # 种/属都不过门；family X = 0.25+0.24+0.21 = 0.70 过门，真值 class0（family X）→ 可用（报科）
    m = hierarchical_usability(
        _from_probs([[0.25, 0.24, 0.21, 0.20, 0.10]]),
        torch.tensor([0]),
        _hier(),
        tau_species=0.9,
        tau_genus=0.6,
        tau_family=0.6,
    )
    assert m.usable_rate == 1.0
    assert m.report_family == 1


def test_fallback_to_bird_always_usable():
    # 全弥散、各层都不过门 → 回退 bird，总可用
    m = hierarchical_usability(
        _from_probs([[0.2, 0.2, 0.2, 0.2, 0.2]]),
        torch.tensor([4]),
        _hier(),
        tau_species=0.9,
        tau_genus=0.9,
        tau_family=0.9,
    )
    assert m.usable_rate == 1.0
    assert m.report_bird == 1


def test_batch_mixed():
    logits = _from_probs(
        [
            [0.9, 0.05, 0.02, 0.02, 0.01],  # 报种正确 (t=0)
            [0.9, 0.05, 0.02, 0.02, 0.01],  # 自信错种 (t=1)
            [0.30, 0.35, 0.20, 0.10, 0.05],  # 回退属正确 (t=0)
        ]
    )
    m = hierarchical_usability(logits, torch.tensor([0, 1, 0]), _hier())
    assert m.n == 3
    assert m.usable_rate == 2 / 3
    assert m.species_correct == 1 and m.critical_error == 1 and m.report_genus == 1
