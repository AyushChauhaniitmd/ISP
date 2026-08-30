from dp_forgetbench.statistics import redundancy_screen


def test_redundancy_screen_requires_five_seeds_for_exploratory_decision() -> None:
    result = redundancy_screen(
        [0.05, 0.04, 0.03, 0.05, 0.04],
        [0.04, 0.03, 0.02, 0.04, 0.03],
        [0.05, 0.05, 0.04, 0.06, 0.05],
    )
    assert result["decision"] == "exploratory_screen_only"
    assert result["marginal_unlearning_benefit"]["n_seeds"] == 5
