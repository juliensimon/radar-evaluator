"""Smoke tests for data extraction functionality."""

import json
from unittest.mock import Mock, mock_open, patch

from radar_evaluator import RadarEvaluator


def test_smoke_data_extraction():
    """Basic smoke test for data extraction functionality."""
    sample_config = {
        "models": {
            "model1": {"name": "Test Model 1", "model_endpoint": "test/model1"},
        },
        "metrics": {
            "accuracy": {"min": 1.0, "max": 5.0},
            "completeness": {"min": 1.0, "max": 5.0},
            "clarity": {"min": 1.0, "max": 5.0},
            "depth": {"min": 1.0, "max": 5.0},
            "overall": {"min": 1.0, "max": 5.0},
        },
        "evaluation": {
            "max_tokens": 1000,
            "temperature": 0.7,
            "evaluator_model": "deepseek/deepseek-coder",
            "evaluation_max_tokens": 500,
            "evaluation_temperature": 0.3,
        },
        "output": {"results_dir": "radar_results", "max_workers": 4},
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(sample_config))):
        with patch.object(RadarEvaluator, "initialize_client", return_value=Mock()):
            evaluator = RadarEvaluator()

        # Test basic score extraction
        evaluation_text = """
        Accuracy: 4.5/5
        Completeness: 3.8/5
        Clarity: 4.2/5
        Depth: 3.9/5
        Overall Quality: 4.1/5
        """

        scores = evaluator.extract_scores_from_evaluation(evaluation_text)

        # Verify all scores are extracted correctly
        assert scores["accuracy"] == 4.5
        assert scores["completeness"] == 3.8
        assert scores["clarity"] == 4.2
        assert scores["depth"] == 3.9
        assert scores["overall"] == 4.1

        # Test score validation
        assert evaluator.validate_scores(scores) is True


def test_smoke_config_loading():
    """Basic smoke test for config loading."""
    sample_config = {
        "models": {"model1": {"name": "Test Model", "model_endpoint": "test/model"}},
        "metrics": {"accuracy": {"min": 1.0, "max": 5.0}},
        "evaluation": {"max_tokens": 1000, "temperature": 0.7},
        "output": {"results_dir": "radar_results"},
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(sample_config))):
        with patch.object(RadarEvaluator, "initialize_client", return_value=Mock()):
            evaluator = RadarEvaluator()
        assert evaluator.config == sample_config


def test_smoke_questions_loading():
    """Basic smoke test for questions loading."""
    sample_config = {
        "models": {"model1": {"name": "Test Model", "model_endpoint": "test/model"}},
        "metrics": {"accuracy": {"min": 1.0, "max": 5.0}},
        "evaluation": {"max_tokens": 1000, "temperature": 0.7},
        "output": {"results_dir": "radar_results"},
    }

    sample_questions = {
        "healthcare": [{"question": "Test question", "context": "Test context"}]
    }

    config_mock = mock_open(read_data=json.dumps(sample_config))
    questions_mock = mock_open(read_data=json.dumps(sample_questions))

    with patch("builtins.open") as mock_open_func:
        mock_open_func.side_effect = [
            config_mock.return_value,
            questions_mock.return_value,
        ]
        with patch.object(RadarEvaluator, "initialize_client", return_value=Mock()):
            evaluator = RadarEvaluator()
        questions = evaluator.load_industry_questions("test_questions.json")
        assert questions == sample_questions
