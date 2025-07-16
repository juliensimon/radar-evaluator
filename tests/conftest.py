"""Pytest configuration and fixtures."""

import json
import os
import tempfile
from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
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


@pytest.fixture
def sample_questions():
    """Provide sample industry questions for testing."""
    return {
        "healthcare": [
            {
                "question": "What are the benefits of telemedicine?",
                "context": "Healthcare technology",
            },
            {
                "question": "How does AI improve patient diagnosis?",
                "context": "Medical AI applications",
            },
        ],
        "technology": [
            {"question": "What is machine learning?", "context": "AI fundamentals"},
            {
                "question": "How do neural networks work?",
                "context": "Deep learning basics",
            },
        ],
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_questions_file(sample_questions):
    """Create a temporary questions file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_questions, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_together_client():
    """Mock Together AI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_evaluation_text():
    """Provide sample evaluation text for testing."""
    return """
    Accuracy: 4.5/5
    Completeness: 3.8/5
    Clarity: 4.2/5
    Depth: 3.9/5
    Overall Quality: 4.1/5

    This is a well-structured response that addresses the question comprehensively.
    """


@pytest.fixture
def sample_scores():
    """Provide sample scores for testing."""
    return {
        "accuracy": 4.5,
        "completeness": 3.8,
        "clarity": 4.2,
        "depth": 3.9,
        "overall": 4.1,
    }


@pytest.fixture
def sample_results():
    """Provide sample evaluation results for testing."""
    return [
        {
            "industry": "healthcare",
            "prompt_idx": 1,
            "question": "What are the benefits of telemedicine?",
            "model_1_response": "Telemedicine offers remote healthcare access...",
            "model_2_response": "Telemedicine provides virtual medical consultations...",
            "model_1_evaluation": "Good response covering key benefits...",
            "model_2_evaluation": "Comprehensive overview of telemedicine...",
            "model_1_scores": {
                "accuracy": 4.0,
                "completeness": 3.5,
                "clarity": 4.2,
                "depth": 3.8,
                "overall": 3.9,
            },
            "model_2_scores": {
                "accuracy": 3.8,
                "completeness": 4.1,
                "clarity": 3.9,
                "depth": 4.0,
                "overall": 4.0,
            },
        }
    ]
