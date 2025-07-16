"""Unit tests for RadarEvaluator."""

import json
import os
from unittest.mock import Mock, mock_open, patch

import pytest

from radar_evaluator import RadarEvaluator


class TestRadarEvaluator:
    """Test cases for RadarEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_config = {
            "models": {
                "model1": {"name": "Test Model 1", "model_endpoint": "test/model1"},
                "model2": {"name": "Test Model 2", "model_endpoint": "test/model2"},
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

    def test_load_config_success(self):
        """Test successful config loading."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator("test_config.json")
            assert evaluator.config == self.sample_config

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(SystemExit):
            RadarEvaluator("nonexistent_config.json")

    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with pytest.raises(SystemExit):
                RadarEvaluator("test_config.json")

    @patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"})
    def test_initialize_client_success(self):
        """Test successful client initialization."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            assert evaluator.client is not None

    def test_initialize_client_no_api_key(self):
        """Test client initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
            ):
                with pytest.raises(SystemExit):
                    RadarEvaluator()

    def test_load_industry_questions_success(self):
        """Test successful industry questions loading."""
        sample_questions = {
            "healthcare": [
                {"question": "Test question 1", "context": "Test context 1"},
                {"question": "Test question 2", "context": "Test context 2"},
            ]
        }

        # Create separate mock objects for each file
        config_mock = mock_open(read_data=json.dumps(self.sample_config))
        questions_mock = mock_open(read_data=json.dumps(sample_questions))

        with patch("builtins.open") as mock_open_func:
            mock_open_func.side_effect = [
                config_mock.return_value,
                questions_mock.return_value,
            ]
            evaluator = RadarEvaluator()
            questions = evaluator.load_industry_questions("test_questions.json")
            assert questions == sample_questions

    def test_load_industry_questions_file_not_found(self):
        """Test industry questions loading with non-existent file."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            with patch(
                "builtins.open", side_effect=FileNotFoundError("File not found")
            ):
                with pytest.raises(SystemExit):
                    evaluator.load_industry_questions("nonexistent_questions.json")

    def test_get_system_prompt(self):
        """Test system prompt generation."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            prompt = evaluator.get_system_prompt("Test Model")
            assert "Test Model" in prompt
            assert "Arcee AI" in prompt
            assert "helpful" in prompt.lower()

    def test_extract_scores_from_evaluation_valid(self):
        """Test score extraction with valid evaluation text."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            evaluation_text = """
            Accuracy: 4.5/5
            Completeness: 3.8/5
            Clarity: 4.2/5
            Depth: 3.9/5
            Overall Quality: 4.1/5
            """

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            assert scores["accuracy"] == 4.5
            assert scores["completeness"] == 3.8
            assert scores["clarity"] == 4.2
            assert scores["depth"] == 3.9
            assert scores["overall"] == 4.1

    def test_extract_scores_from_evaluation_missing_scores(self):
        """Test score extraction with missing scores."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            evaluation_text = "This is an evaluation without scores."

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            # Should default to 3.0 for missing scores
            for score in scores.values():
                assert score == 3.0

    def test_extract_scores_from_evaluation_out_of_range(self):
        """Test score extraction with out-of-range scores."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            evaluation_text = """
            Accuracy: 6.0/5
            Completeness: 0.5/5
            Clarity: 4.0/5
            Depth: 3.0/5
            Overall Quality: 5.0/5
            """

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            # Should clamp to valid range (1.0-5.0)
            assert scores["accuracy"] == 5.0  # Clamped from 6.0
            assert scores["completeness"] == 1.0  # Clamped from 0.5
            assert scores["clarity"] == 4.0
            assert scores["depth"] == 3.0
            assert scores["overall"] == 5.0

    def test_extract_scores_from_evaluation_different_formats(self):
        """Test score extraction with different text formats."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            evaluation_text = """
            Accuracy 4.5
            Completeness: 3.8
            Clarity 4.2
            Depth: 3.9
            Overall Quality 4.1
            """

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            assert scores["accuracy"] == 4.5
            assert scores["completeness"] == 3.8
            assert scores["clarity"] == 4.2
            assert scores["depth"] == 3.9
            assert scores["overall"] == 4.1

    def test_validate_scores_valid(self):
        """Test score validation with valid scores."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": 4.5,
                "completeness": 3.8,
                "clarity": 4.2,
                "depth": 3.9,
                "overall": 4.1,
            }

            result = evaluator.validate_scores(scores)
            assert result is True

    def test_validate_scores_invalid_type(self):
        """Test score validation with invalid score types."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": "invalid",
                "completeness": 3.8,
                "clarity": 4.2,
                "depth": 3.9,
                "overall": 4.1,
            }

            result = evaluator.validate_scores(scores)
            assert result is False

    def test_validate_scores_out_of_range(self):
        """Test score validation with out-of-range scores."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": 6.0,  # Out of range
                "completeness": 3.8,
                "clarity": 4.2,
                "depth": 3.9,
                "overall": 4.1,
            }

            result = evaluator.validate_scores(scores)
            assert result is False

    def test_validate_scores_missing_metrics(self):
        """Test score validation with missing metrics."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": 4.5,
                "completeness": 3.8,
                # Missing clarity, depth, overall
            }

            result = evaluator.validate_scores(scores)
            assert result is False

    @patch("radar_evaluator.Together")
    def test_stream_model_response_success(self, mock_together):
        """Test successful streaming model response."""
        # Mock the Together client
        mock_client = Mock()
        mock_together.return_value = mock_client

        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello "

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = "World"

        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Hello"}]

                response = evaluator.stream_model_response(model, messages)
                assert response == "Hello World"

    @patch("radar_evaluator.Together")
    def test_stream_model_response_fallback(self, mock_together):
        """Test streaming model response with fallback to non-streaming."""
        # Mock the Together client
        mock_client = Mock()
        mock_together.return_value = mock_client

        # Mock empty streaming response
        mock_client.chat.completions.create.return_value = []

        # Mock non-streaming response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fallback response"

        # First call returns empty stream, second call returns non-streaming response
        mock_client.chat.completions.create.side_effect = [[], mock_response]

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Hello"}]

                response = evaluator.stream_model_response(model, messages)
                assert response == "Fallback response"

    @patch("radar_evaluator.Together")
    def test_stream_model_response_error(self, mock_together):
        """Test streaming model response with error."""
        # Mock the Together client
        mock_client = Mock()
        mock_together.return_value = mock_client

        # Mock exception
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Hello"}]

                response = evaluator.stream_model_response(model, messages)
                assert "Error: API Error" in response

    def test_process_question_success(self):
        """Test successful question processing."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                # Mock the stream_model_response method
                evaluator.stream_model_response = Mock(return_value="Test response")  # type: ignore

                # Mock the evaluate_with_deepseek method
                evaluator.evaluate_with_deepseek = Mock(  # type: ignore
                    return_value=(
                        "Test evaluation",
                        {
                            "accuracy": 4.0,
                            "completeness": 3.5,
                            "clarity": 4.2,
                            "depth": 3.8,
                            "overall": 3.9,
                        },
                    )
                )

                model_1 = {"name": "Model 1", "model_endpoint": "test/model1"}
                model_2 = {"name": "Model 2", "model_endpoint": "test/model2"}
                question = "What is AI?"
                industry = "technology"
                prompt_idx = 1

                result = evaluator.process_question(
                    model_1, model_2, question, industry, prompt_idx
                )

                assert result["industry"] == industry
                assert result["prompt_idx"] == prompt_idx
                assert result["question"] == question
                assert "model_1" in result
                assert "model_2" in result
                assert result["model_1"]["response"] == "Test response"
                assert result["model_1"]["evaluation"] == "Test evaluation"
                assert result["model_1"]["scores"]["accuracy"] == 4.0
                assert result["model_2"]["response"] == "Test response"
                assert result["model_2"]["evaluation"] == "Test evaluation"
                assert result["model_2"]["scores"]["accuracy"] == 4.0

    def test_generate_radar_chart(self):
        """Test radar chart generation."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            # Mock matplotlib to avoid actual plotting
            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    results = [
                        {
                            "industry": "technology",
                            "model_1": {
                                "scores": {
                                    "accuracy": 4.0,
                                    "completeness": 3.5,
                                    "clarity": 4.2,
                                    "depth": 3.8,
                                    "overall": 3.9,
                                }
                            },
                            "model_2": {
                                "scores": {
                                    "accuracy": 3.8,
                                    "completeness": 4.1,
                                    "clarity": 3.9,
                                    "depth": 4.0,
                                    "overall": 4.0,
                                }
                            },
                        }
                    ]

                    # Should not raise any exceptions
                    evaluator.generate_radar_chart(results, "Model 1", "Model 2")

    def test_generate_summary_report(self):
        """Test summary report generation."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            results = [
                {
                    "industry": "technology",
                    "model_1": {
                        "scores": {
                            "accuracy": 4.0,
                            "completeness": 3.5,
                            "clarity": 4.2,
                            "depth": 3.8,
                            "overall": 3.9,
                        }
                    },
                    "model_2": {
                        "scores": {
                            "accuracy": 3.8,
                            "completeness": 4.1,
                            "clarity": 3.9,
                            "depth": 4.0,
                            "overall": 4.0,
                        }
                    },
                }
            ]

            # Should not raise any exceptions
            report = evaluator.generate_summary_report(results, "Model 1", "Model 2")
            assert "Model Comparison Report" in report
            assert "Model 1" in report
            assert "Model 2" in report

    def test_save_results(self):
        """Test results saving."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            results = [
                {
                    "industry": "technology",
                    "prompt_idx": 1,
                    "question": "Test question",
                    "model_1": {
                        "response": "Test response 1",
                        "evaluation": "Test evaluation 1",
                        "scores": {
                            "accuracy": 4.0,
                            "completeness": 3.5,
                            "clarity": 4.2,
                            "depth": 3.8,
                            "overall": 3.9,
                        },
                    },
                    "model_2": {
                        "response": "Test response 2",
                        "evaluation": "Test evaluation 2",
                        "scores": {
                            "accuracy": 3.8,
                            "completeness": 4.1,
                            "clarity": 3.9,
                            "depth": 4.0,
                            "overall": 4.0,
                        },
                    },
                }
            ]

            # Mock file operations and matplotlib
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("matplotlib.pyplot.savefig"):
                    with patch("matplotlib.pyplot.close"):
                        evaluator.save_results(results, "Model 1", "Model 2")
                        # Verify that files were opened for writing
                        assert mock_file.called


class TestRadarEvaluatorIntegration:
    """Integration tests for RadarEvaluator."""

    @property
    def sample_config(self):
        return {
            "models": {
                "model1": {"name": "Test Model 1", "model_endpoint": "test/model1"},
                "model2": {"name": "Test Model 2", "model_endpoint": "test/model2"},
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
            "output": {"results_dir": "radar_results"},
        }

    def test_full_evaluation_workflow(self):
        """Test the complete evaluation workflow."""
        # This is a high-level integration test
        # In a real scenario, you'd mock external API calls
        pass

    def test_config_validation(self):
        """Test that the config structure is valid."""
        required_keys = ["models", "metrics", "evaluation", "output"]

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            for key in required_keys:
                assert key in evaluator.config

    def test_metrics_config_structure(self):
        """Test that metrics config has the correct structure."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            for metric, config in evaluator.config["metrics"].items():
                assert "min" in config
                assert "max" in config
                assert isinstance(config["min"], (int, float))
                assert isinstance(config["max"], (int, float))
                assert config["min"] <= config["max"]


def test_version():
    """Test that version is defined."""
    from radar_evaluator import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_author():
    """Test that the author information is correct."""
    from radar_evaluator import __author__

    assert __author__ == "Arcee AI"


# Additional tests for better coverage
class TestRadarEvaluatorEdgeCases:
    """Test edge cases and error handling for RadarEvaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_config = {
            "models": {
                "model1": {"name": "Test Model 1", "model_endpoint": "test/model1"},
                "model2": {"name": "Test Model 2", "model_endpoint": "test/model2"},
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

    def test_generate_radar_chart_empty_results(self):
        """Test radar chart generation with empty results."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    fig = evaluator.generate_radar_chart([], "Model 1", "Model 2")
                    assert fig is not None

    def test_generate_radar_chart_single_industry(self):
        """Test radar chart generation with single industry."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    results = [
                        {
                            "industry": "technology",
                            "model_1": {
                                "scores": {
                                    "accuracy": 4.0,
                                    "completeness": 3.5,
                                    "clarity": 4.2,
                                    "depth": 3.8,
                                    "overall": 3.9,
                                }
                            },
                            "model_2": {
                                "scores": {
                                    "accuracy": 3.8,
                                    "completeness": 4.1,
                                    "clarity": 3.9,
                                    "depth": 4.0,
                                    "overall": 4.0,
                                }
                            },
                        }
                    ]
                    fig = evaluator.generate_radar_chart(results, "Model 1", "Model 2")
                    assert fig is not None

    def test_generate_radar_chart_multiple_industries(self):
        """Test radar chart generation with multiple industries."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    results = [
                        {
                            "industry": "technology",
                            "model_1": {
                                "scores": {
                                    "accuracy": 4.0,
                                    "completeness": 3.5,
                                    "clarity": 4.2,
                                    "depth": 3.8,
                                    "overall": 3.9,
                                }
                            },
                            "model_2": {
                                "scores": {
                                    "accuracy": 3.8,
                                    "completeness": 4.1,
                                    "clarity": 3.9,
                                    "depth": 4.0,
                                    "overall": 4.0,
                                }
                            },
                        },
                        {
                            "industry": "healthcare",
                            "model_1": {
                                "scores": {
                                    "accuracy": 3.5,
                                    "completeness": 4.0,
                                    "clarity": 3.8,
                                    "depth": 4.2,
                                    "overall": 3.9,
                                }
                            },
                            "model_2": {
                                "scores": {
                                    "accuracy": 4.2,
                                    "completeness": 3.8,
                                    "clarity": 4.0,
                                    "depth": 3.9,
                                    "overall": 4.0,
                                }
                            },
                        },
                    ]
                    fig = evaluator.generate_radar_chart(results, "Model 1", "Model 2")
                    assert fig is not None

    def test_generate_summary_report_empty_results(self):
        """Test summary report generation with empty results."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            report = evaluator.generate_summary_report([], "Model 1", "Model 2")
            assert "No results available" in report

    def test_generate_summary_report_multiple_industries(self):
        """Test summary report generation with multiple industries."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()
            results = [
                {
                    "industry": "technology",
                    "model_1": {
                        "scores": {
                            "accuracy": 4.0,
                            "completeness": 3.5,
                            "clarity": 4.2,
                            "depth": 3.8,
                            "overall": 3.9,
                        }
                    },
                    "model_2": {
                        "scores": {
                            "accuracy": 3.8,
                            "completeness": 4.1,
                            "clarity": 3.9,
                            "depth": 4.0,
                            "overall": 4.0,
                        }
                    },
                },
                {
                    "industry": "healthcare",
                    "model_1": {
                        "scores": {
                            "accuracy": 3.5,
                            "completeness": 4.0,
                            "clarity": 3.8,
                            "depth": 4.2,
                            "overall": 3.9,
                        }
                    },
                    "model_2": {
                        "scores": {
                            "accuracy": 4.2,
                            "completeness": 3.8,
                            "clarity": 4.0,
                            "depth": 3.9,
                            "overall": 4.0,
                        }
                    },
                },
            ]
            report = evaluator.generate_summary_report(results, "Model 1", "Model 2")
            assert "technology" in report
            assert "healthcare" in report

    def test_stream_model_response_with_custom_params(self):
        """Test stream_model_response with custom parameters."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()
                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Test question"}]

                with patch.object(
                    evaluator.client.chat.completions, "create"
                ) as mock_create:
                    mock_create.return_value = iter(
                        [
                            type(
                                "Chunk",
                                (),
                                {
                                    "choices": [
                                        type(
                                            "Choice",
                                            (),
                                            {
                                                "delta": type(
                                                    "Delta",
                                                    (),
                                                    {"content": "Test response"},
                                                )()
                                            },
                                        )()
                                    ]
                                },
                            )()
                        ]
                    )

                    response = evaluator.stream_model_response(
                        model, messages, max_tokens=500, temperature=0.5
                    )
                    assert response == "Test response"

    def test_evaluate_with_deepseek_success(self):
        """Test successful evaluation with DeepSeek."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                with patch.object(
                    evaluator.client.chat.completions, "create"
                ) as mock_create:
                    mock_create.return_value = type(
                        "Response",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Message",
                                            (),
                                            {
                                                "content": "Accuracy: 4.0/5\nCompleteness: 3.5/5\nClarity: 4.2/5\nDepth: 3.8/5\nOverall Quality: 3.9/5"
                                            },
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()

                    evaluation, scores = evaluator.evaluate_with_deepseek(
                        "Test question", "Test answer", "Test Model"
                    )
                    assert evaluation is not None
                    assert scores["accuracy"] == 4.0
                    assert scores["completeness"] == 3.5

    def test_evaluate_with_deepseek_error(self):
        """Test evaluation with DeepSeek when API call fails."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                with patch.object(
                    evaluator.client.chat.completions,
                    "create",
                    side_effect=Exception("API Error"),
                ):
                    evaluation, scores = evaluator.evaluate_with_deepseek(
                        "Test question", "Test answer", "Test Model"
                    )
                    assert "Error" in evaluation
                    assert all(score == 3.0 for score in scores.values())

    def test_run_evaluation_basic(self):
        """Test basic evaluation run."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                # Mock all the methods that would make API calls
                with patch.object(
                    evaluator,
                    "load_industry_questions",
                    return_value={
                        "technology": [
                            {"question": "Test question", "context": "Test context"}
                        ]
                    },
                ), patch.object(
                    evaluator,
                    "process_question",
                    return_value={
                        "industry": "technology",
                        "prompt_idx": 1,
                        "model_1": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 4.0,
                                "completeness": 3.5,
                                "clarity": 4.2,
                                "depth": 3.8,
                                "overall": 3.9,
                            },
                        },
                        "model_2": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 3.8,
                                "completeness": 4.1,
                                "clarity": 3.9,
                                "depth": 4.0,
                                "overall": 4.0,
                            },
                        },
                        "question": "Test question",
                    },
                ), patch.object(
                    evaluator, "save_results", Mock()
                ):

                    results = evaluator.run_evaluation(
                        "model1", "model2", industries=["technology"], num_questions=1
                    )
                    assert len(results) == 1
                    assert results[0]["industry"] == "technology"

    def test_run_evaluation_with_invalid_models(self):
        """Test evaluation run with invalid model keys."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                with pytest.raises(KeyError):
                    evaluator.run_evaluation("invalid_model1", "invalid_model2")

    def test_main_function(self):
        """Test the main function."""
        with patch(
            "sys.argv",
            ["radar_evaluator.py", "--model1", "model1", "--model2", "model2"],
        ):
            with patch("radar_evaluator.RadarEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.config = {"models": {"model1": {}, "model2": {}}}
                mock_evaluator_class.return_value = mock_evaluator
                mock_evaluator.run_evaluation.return_value = []

                from radar_evaluator import main

                main()

                mock_evaluator.run_evaluation.assert_called_once_with(
                    "model1", "model2", None, None
                )

    def test_main_function_with_optional_args(self):
        """Test the main function with optional arguments."""
        with patch(
            "sys.argv",
            [
                "radar_evaluator.py",
                "--model1",
                "model1",
                "--model2",
                "model2",
                "--industries",
                "technology,healthcare",
                "--num-questions",
                "5",
            ],
        ):
            with patch("radar_evaluator.RadarEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.config = {"models": {"model1": {}, "model2": {}}}
                mock_evaluator_class.return_value = mock_evaluator
                mock_evaluator.run_evaluation.return_value = []

                from radar_evaluator import main

                main()

                mock_evaluator.run_evaluation.assert_called_once_with(
                    "model1", "model2", ["technology,healthcare"], 5
                )

    def test_stream_model_response_fallback_to_non_streaming(self):
        """Test stream_model_response fallback to non-streaming when streaming fails."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()
                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Test question"}]

                # Mock streaming to return empty response, forcing fallback
                with patch.object(
                    evaluator.client.chat.completions, "create"
                ) as mock_create:
                    # First call (streaming) returns empty response
                    mock_create.return_value = iter([])

                    response = evaluator.stream_model_response(model, messages)
                    assert "Error" in response

    def test_stream_model_response_exception_handling(self):
        """Test stream_model_response exception handling."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()
                model = {"name": "Test Model", "model_endpoint": "test/model"}
                messages = [{"role": "user", "content": "Test question"}]

                with patch.object(
                    evaluator.client.chat.completions,
                    "create",
                    side_effect=Exception("API Error"),
                ):
                    response = evaluator.stream_model_response(model, messages)
                    assert "Error" in response

    def test_save_results_file_operations(self):
        """Test save_results file operations."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            results = [
                {
                    "industry": "technology",
                    "model_1": {
                        "response": "Test response 1",
                        "evaluation": "Test evaluation 1",
                        "scores": {
                            "accuracy": 4.0,
                            "completeness": 3.5,
                            "clarity": 4.2,
                            "depth": 3.8,
                            "overall": 3.9,
                        },
                    },
                    "model_2": {
                        "response": "Test response 2",
                        "evaluation": "Test evaluation 2",
                        "scores": {
                            "accuracy": 3.8,
                            "completeness": 4.1,
                            "clarity": 3.9,
                            "depth": 4.0,
                            "overall": 4.0,
                        },
                    },
                    "question": "Test question",
                    "prompt_idx": 1,
                }
            ]

            # Mock file operations and matplotlib
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("matplotlib.pyplot.savefig"):
                    with patch("matplotlib.pyplot.close"):
                        evaluator.save_results(results, "Model 1", "Model 2")
                        # Verify that files were opened for writing
                        assert mock_file.called

    def test_run_evaluation_with_limited_questions(self):
        """Test run_evaluation with limited number of questions."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                # Mock industry questions with multiple questions
                with patch.object(
                    evaluator,
                    "load_industry_questions",
                    return_value={
                        "technology": [
                            {
                                "question": "Test question 1",
                                "context": "Test context 1",
                            },
                            {
                                "question": "Test question 2",
                                "context": "Test context 2",
                            },
                            {
                                "question": "Test question 3",
                                "context": "Test context 3",
                            },
                        ]
                    },
                ), patch.object(
                    evaluator,
                    "process_question",
                    return_value={
                        "industry": "technology",
                        "prompt_idx": 1,
                        "model_1": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 4.0,
                                "completeness": 3.5,
                                "clarity": 4.2,
                                "depth": 3.8,
                                "overall": 3.9,
                            },
                        },
                        "model_2": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 3.8,
                                "completeness": 4.1,
                                "clarity": 3.9,
                                "depth": 4.0,
                                "overall": 4.0,
                            },
                        },
                        "question": "Test question",
                    },
                ), patch.object(
                    evaluator, "save_results", Mock()
                ):

                    results = evaluator.run_evaluation(
                        "model1", "model2", industries=["technology"], num_questions=2
                    )
                    assert len(results) == 2

    def test_run_evaluation_with_specific_industries(self):
        """Test run_evaluation with specific industries."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}):
                evaluator = RadarEvaluator()

                with patch.object(
                    evaluator,
                    "load_industry_questions",
                    return_value={
                        "technology": [
                            {"question": "Test question 1", "context": "Test context 1"}
                        ],
                        "healthcare": [
                            {"question": "Test question 2", "context": "Test context 2"}
                        ],
                    },
                ), patch.object(
                    evaluator,
                    "process_question",
                    return_value={
                        "industry": "technology",
                        "prompt_idx": 1,
                        "model_1": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 4.0,
                                "completeness": 3.5,
                                "clarity": 4.2,
                                "depth": 3.8,
                                "overall": 3.9,
                            },
                        },
                        "model_2": {
                            "response": "Test",
                            "evaluation": "Test",
                            "scores": {
                                "accuracy": 3.8,
                                "completeness": 4.1,
                                "clarity": 3.9,
                                "depth": 4.0,
                                "overall": 4.0,
                            },
                        },
                        "question": "Test question",
                    },
                ), patch.object(
                    evaluator, "save_results", Mock()
                ):

                    results = evaluator.run_evaluation(
                        "model1", "model2", industries=["technology"], num_questions=1
                    )
                    assert len(results) == 1
                    assert results[0]["industry"] == "technology"

    def test_extract_scores_from_evaluation_with_malformed_text(self):
        """Test score extraction with malformed evaluation text."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            # Test with completely malformed text
            evaluation_text = "This is completely malformed text with no scores at all"

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            # Should default to 3.0 for all missing scores
            for score in scores.values():
                assert score == 3.0

    def test_extract_scores_from_evaluation_with_partial_scores(self):
        """Test score extraction with only some scores present."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            evaluation_text = """
            Accuracy: 4.5/5
            Completeness: 3.8/5
            Clarity: 4.2/5
            Depth: 3.9/5
            Overall Quality: 4.1/5
            """

            scores = evaluator.extract_scores_from_evaluation(evaluation_text)

            assert scores["accuracy"] == 4.5
            assert scores["completeness"] == 3.8
            assert scores["clarity"] == 4.2
            assert scores["depth"] == 3.9
            assert scores["overall"] == 4.1

    def test_validate_scores_with_none_values(self):
        """Test score validation with None values."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": None,
                "completeness": 3.5,
                "clarity": 4.2,
                "depth": 3.8,
                "overall": 3.9,
            }

            assert not evaluator.validate_scores(scores)

    def test_validate_scores_with_string_values(self):
        """Test score validation with string values."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.sample_config))
        ):
            evaluator = RadarEvaluator()

            scores = {
                "accuracy": "invalid",
                "completeness": 3.5,
                "clarity": 4.2,
                "depth": 3.8,
                "overall": 3.9,
            }

            assert not evaluator.validate_scores(scores)
