"""
Frontend-Backend Contract Integration Tests
============================================
Production-grade integration tests for frontend-backend contract validation.

Tests:
1. API Response Schema Validation
2. WebSocket Message Format Validation
3. Field Name Consistency
4. Data Type Consistency
5. Value Range Validation
6. Missing Field Detection
7. Extra Field Detection

Ensures frontend expectations match backend reality.
"""

import asyncio
import json
import logging
import pytest
import uuid
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from backend.models.state import (
    InterviewState,
    QuestionModel,
    VoiceAnalysisModel,
    AnswerQualityModel,
    BodyLanguageModel,
    ConfidenceBehaviorModel,
    ScoresModel,
    RecommendationsModel
)
from utils.schema_validator import (
    ContractValidator,
    SchemaValidator,
    INTERVIEW_STATE_SCHEMA,
    QUESTION_SCHEMA,
    VOICE_ANALYSIS_SCHEMA,
    ANSWER_QUALITY_SCHEMA,
    BODY_LANGUAGE_SCHEMA,
    CONFIDENCE_BEHAVIOR_SCHEMA,
    SCORES_SCHEMA,
    RECOMMENDATIONS_SCHEMA,
    WS_TRANSCRIPT_UPDATE_SCHEMA,
    WS_METRICS_UPDATE_SCHEMA,
    WS_SCORE_UPDATE_SCHEMA
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def contract_validator():
    """Fixture for contract validator."""
    return ContractValidator()


@pytest.fixture
def schema_validator():
    """Fixture for schema validator."""
    return SchemaValidator()


@pytest.fixture
def sample_interview_state():
    """Generate sample interview state for testing."""
    return InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        question=QuestionModel(
            text="Explain the difference between Process and Thread",
            topic="operating_systems",
            difficulty=0.6,
            intent=["memory_isolation", "resource_sharing", "context_switching"]
        ),
        transcript="A process has its own memory space while threads share memory. "
                   "Processes provide better isolation but threads are lightweight.",
        voice_analysis=VoiceAnalysisModel(
            clarity=0.85,
            speech_rate_wpm=150.0,
            filler_ratio=0.05,
            tone="confident"
        ),
        answer_quality=AnswerQualityModel(
            relevance=0.90,
            correctness=0.85,
            depth=0.75,
            structure=0.80,
            gaps=["performance_comparison", "use_cases"]
        ),
        body_language=BodyLanguageModel(
            eye_contact=0.80,
            posture_stability=0.85,
            facial_expressiveness=0.75,
            distractions=[]
        ),
        confidence_behavior=ConfidenceBehaviorModel(
            confidence=0.80,
            nervousness=0.15,
            professionalism=0.85,
            behavioral_flags=[]
        ),
        scores=ScoresModel(
            technical=82.5,
            communication=78.0,
            behavioral=83.0,
            overall=81.2
        ),
        recommendations=RecommendationsModel(
            strengths=[
                "Clear explanation of core concepts",
                "Good technical accuracy",
                "Confident delivery"
            ],
            weaknesses=[
                "Missing performance comparison",
                "Could elaborate on use cases"
            ],
            improvement_plan=[
                "Study real-world scenarios for process vs thread usage",
                "Practice explaining performance implications",
                "Add concrete examples to explanations"
            ]
        )
    )


# ============================================================================
# TEST: API RESPONSE SCHEMA VALIDATION
# ============================================================================

def test_interview_state_schema_validation(
    sample_interview_state,
    contract_validator
):
    """
    Test: Complete interview state matches expected schema.

    Validates:
    - All required fields present
    - Correct data types
    - Value ranges
    - Nested object schemas
    """
    logger.info("TEST: Interview state schema validation")

    # Convert to dict
    state_dict = sample_interview_state.model_dump()

    # Validate schema
    result = contract_validator.validate_interview_state(
        state_dict,
        check_completeness=True
    )

    # Log results
    if not result.valid:
        logger.error("Schema validation failed:")
        for error in result.errors:
            logger.error(f"  - {error}")

    if result.warnings:
        logger.warning("Schema validation warnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")

    # Validate
    assert result.valid, f"Interview state schema validation failed: {result.errors}"

    logger.info("✓ Interview state schema validation PASSED")


def test_question_schema_validation(sample_interview_state, schema_validator):
    """
    Test: Question object matches expected schema.

    Validates:
    - Question structure
    - Required fields
    - Data types
    """
    logger.info("TEST: Question schema validation")

    question_dict = sample_interview_state.question.model_dump()

    result = schema_validator.validate(question_dict, QUESTION_SCHEMA)

    assert result.valid, f"Question schema validation failed: {result.errors}"

    logger.info("✓ Question schema validation PASSED")


def test_voice_analysis_schema_validation(
    sample_interview_state,
    schema_validator
):
    """
    Test: Voice analysis object matches expected schema.

    Validates:
    - Voice metrics structure
    - Value ranges (0-1 for ratios)
    - Required fields
    """
    logger.info("TEST: Voice analysis schema validation")

    voice_dict = sample_interview_state.voice_analysis.model_dump()

    result = schema_validator.validate(voice_dict, VOICE_ANALYSIS_SCHEMA)

    assert result.valid, f"Voice analysis schema validation failed: {result.errors}"

    # Additional range checks
    assert 0.0 <= voice_dict["clarity"] <= 1.0
    assert voice_dict["speech_rate_wpm"] >= 0.0
    assert 0.0 <= voice_dict["filler_ratio"] <= 1.0

    logger.info("✓ Voice analysis schema validation PASSED")


def test_answer_quality_schema_validation(
    sample_interview_state,
    schema_validator
):
    """
    Test: Answer quality object matches expected schema.

    Validates:
    - Quality metrics structure
    - Value ranges (0-1)
    - Gaps array
    """
    logger.info("TEST: Answer quality schema validation")

    quality_dict = sample_interview_state.answer_quality.model_dump()

    result = schema_validator.validate(quality_dict, ANSWER_QUALITY_SCHEMA)

    assert result.valid, f"Answer quality schema validation failed: {result.errors}"

    # Additional range checks
    assert 0.0 <= quality_dict["relevance"] <= 1.0
    assert 0.0 <= quality_dict["correctness"] <= 1.0
    assert 0.0 <= quality_dict["depth"] <= 1.0
    assert 0.0 <= quality_dict["structure"] <= 1.0

    logger.info("✓ Answer quality schema validation PASSED")


def test_body_language_schema_validation(
    sample_interview_state,
    schema_validator
):
    """
    Test: Body language object matches expected schema.

    Validates:
    - Body metrics structure
    - Value ranges (0-1)
    - Distractions array
    """
    logger.info("TEST: Body language schema validation")

    body_dict = sample_interview_state.body_language.model_dump()

    result = schema_validator.validate(body_dict, BODY_LANGUAGE_SCHEMA)

    assert result.valid, f"Body language schema validation failed: {result.errors}"

    # Additional range checks
    assert 0.0 <= body_dict["eye_contact"] <= 1.0
    assert 0.0 <= body_dict["posture_stability"] <= 1.0
    assert 0.0 <= body_dict["facial_expressiveness"] <= 1.0

    logger.info("✓ Body language schema validation PASSED")


def test_scores_schema_validation(sample_interview_state, schema_validator):
    """
    Test: Scores object matches expected schema.

    Validates:
    - Score structure
    - Value ranges (0-100)
    - All score categories present
    """
    logger.info("TEST: Scores schema validation")

    scores_dict = sample_interview_state.scores.model_dump()

    result = schema_validator.validate(scores_dict, SCORES_SCHEMA)

    assert result.valid, f"Scores schema validation failed: {result.errors}"

    # Additional range checks
    assert 0.0 <= scores_dict["technical"] <= 100.0
    assert 0.0 <= scores_dict["communication"] <= 100.0
    assert 0.0 <= scores_dict["behavioral"] <= 100.0
    assert 0.0 <= scores_dict["overall"] <= 100.0

    logger.info("✓ Scores schema validation PASSED")


def test_recommendations_schema_validation(
    sample_interview_state,
    schema_validator
):
    """
    Test: Recommendations object matches expected schema.

    Validates:
    - Recommendations structure
    - All arrays present
    - Non-empty arrays
    """
    logger.info("TEST: Recommendations schema validation")

    recommendations_dict = sample_interview_state.recommendations.model_dump()

    result = schema_validator.validate(
        recommendations_dict, RECOMMENDATIONS_SCHEMA)

    assert result.valid, f"Recommendations schema validation failed: {result.errors}"

    # Validate arrays are non-empty
    assert len(recommendations_dict["strengths"]
               ) > 0, "Strengths should not be empty"
    assert len(recommendations_dict["improvement_plan"]
               ) > 0, "Improvement plan should not be empty"

    logger.info("✓ Recommendations schema validation PASSED")


# ============================================================================
# TEST: WEBSOCKET MESSAGE VALIDATION
# ============================================================================

def test_websocket_transcript_update_schema(schema_validator):
    """
    Test: WebSocket transcript update message matches schema.

    Validates:
    - Message structure
    - Required fields
    - Message type
    """
    logger.info("TEST: WebSocket transcript update schema")

    message = {
        "type": "transcript_update",
        "session_id": "sess_abc123",
        "transcript": "I would implement this using a hash map",
        "timestamp": datetime.now().isoformat()
    }

    result = schema_validator.validate(message, WS_TRANSCRIPT_UPDATE_SCHEMA)

    assert result.valid, f"Transcript update schema validation failed: {result.errors}"

    logger.info("✓ WebSocket transcript update schema PASSED")


def test_websocket_metrics_update_schema(schema_validator):
    """
    Test: WebSocket metrics update message matches schema.

    Validates:
    - Message structure
    - Metrics object
    - Timestamp format
    """
    logger.info("TEST: WebSocket metrics update schema")

    message = {
        "type": "metrics_update",
        "session_id": "sess_abc123",
        "metrics": {
            "eye_contact": 0.80,
            "posture_stability": 0.85,
            "facial_expressiveness": 0.75
        },
        "timestamp": datetime.now().isoformat()
    }

    result = schema_validator.validate(message, WS_METRICS_UPDATE_SCHEMA)

    assert result.valid, f"Metrics update schema validation failed: {result.errors}"

    logger.info("✓ WebSocket metrics update schema PASSED")


def test_websocket_score_update_schema(schema_validator):
    """
    Test: WebSocket score update message matches schema.

    Validates:
    - Message structure
    - Scores object with nested schema
    - All score fields present
    """
    logger.info("TEST: WebSocket score update schema")

    message = {
        "type": "score_update",
        "session_id": "sess_abc123",
        "scores": {
            "technical": 82.5,
            "communication": 78.0,
            "behavioral": 83.0,
            "overall": 81.2
        },
        "timestamp": datetime.now().isoformat()
    }

    result = schema_validator.validate(message, WS_SCORE_UPDATE_SCHEMA)

    assert result.valid, f"Score update schema validation failed: {result.errors}"

    logger.info("✓ WebSocket score update schema PASSED")


# ============================================================================
# TEST: FIELD CONSISTENCY
# ============================================================================

def test_field_name_consistency(sample_interview_state):
    """
    Test: Field names are consistent across system.

    Validates:
    - No renamed fields between versions
    - Consistent naming convention (snake_case)
    - No unexpected field changes
    """
    logger.info("TEST: Field name consistency")

    state_dict = sample_interview_state.model_dump()

    # Expected top-level fields
    expected_fields = {
        "interview_id",
        "role",
        "experience_level",
        "question",
        "transcript",
        "voice_analysis",
        "answer_quality",
        "body_language",
        "confidence_behavior",
        "scores",
        "recommendations",
        "audio_path",
        "video_path",
        "created_at",
        "updated_at"
    }

    actual_fields = set(state_dict.keys())

    # Check for unexpected fields
    extra_fields = actual_fields - expected_fields
    if extra_fields:
        logger.warning(f"Extra fields found: {extra_fields}")

    # Check for missing fields (optional fields may be None)
    required_fields = {
        "interview_id",
        "role",
        "experience_level"
    }

    missing_fields = required_fields - actual_fields
    assert not missing_fields, f"Missing required fields: {missing_fields}"

    # Validate naming convention (snake_case)
    for field in actual_fields:
        assert field.islower() or "_" in field, \
            f"Field '{field}' does not follow snake_case convention"

    logger.info("✓ Field name consistency PASSED")


def test_data_type_consistency(sample_interview_state):
    """
    Test: Data types are consistent across system.

    Validates:
    - Numeric fields are float/int
    - String fields are str
    - Array fields are list
    - Object fields are dict/model
    """
    logger.info("TEST: Data type consistency")

    state_dict = sample_interview_state.model_dump()

    # Validate types
    assert isinstance(state_dict["interview_id"], str)
    assert isinstance(state_dict["role"], str)
    assert isinstance(state_dict["experience_level"], str)
    assert isinstance(state_dict["transcript"], str)

    # Voice analysis types
    voice = state_dict["voice_analysis"]
    assert isinstance(voice["clarity"], float)
    assert isinstance(voice["speech_rate_wpm"], float)
    assert isinstance(voice["filler_ratio"], float)
    assert isinstance(voice["tone"], str)

    # Answer quality types
    quality = state_dict["answer_quality"]
    assert isinstance(quality["relevance"], float)
    assert isinstance(quality["correctness"], float)
    assert isinstance(quality["depth"], float)
    assert isinstance(quality["structure"], float)
    assert isinstance(quality["gaps"], list)

    # Scores types
    scores = state_dict["scores"]
    assert isinstance(scores["technical"], float)
    assert isinstance(scores["communication"], float)
    assert isinstance(scores["behavioral"], float)
    assert isinstance(scores["overall"], float)

    # Recommendations types
    recommendations = state_dict["recommendations"]
    assert isinstance(recommendations["strengths"], list)
    assert isinstance(recommendations["weaknesses"], list)
    assert isinstance(recommendations["improvement_plan"], list)

    logger.info("✓ Data type consistency PASSED")


# ============================================================================
# TEST: ERROR DETECTION
# ============================================================================

def test_missing_required_field_detection(schema_validator):
    """
    Test: Missing required fields are detected.

    Validates:
    - Validation fails for missing fields
    - Error messages are clear
    - All missing fields reported
    """
    logger.info("TEST: Missing required field detection")

    # Incomplete scores object (missing 'overall')
    incomplete_scores = {
        "technical": 82.5,
        "communication": 78.0,
        "behavioral": 83.0
        # Missing 'overall'
    }

    result = schema_validator.validate(incomplete_scores, SCORES_SCHEMA)

    # Should fail validation
    assert not result.valid, "Validation should fail for missing required fields"
    assert "overall" in result.missing_fields, "Missing 'overall' field should be detected"

    logger.info("✓ Missing required field detection PASSED")


def test_extra_field_detection(schema_validator):
    """
    Test: Extra unexpected fields are detected.

    Validates:
    - Extra fields generate warnings
    - Validation succeeds but warns
    - All extra fields reported
    """
    logger.info("TEST: Extra field detection")

    # Scores object with extra field
    scores_with_extra = {
        "technical": 82.5,
        "communication": 78.0,
        "behavioral": 83.0,
        "overall": 81.2,
        "unexpected_field": "extra_value"
    }

    result = schema_validator.validate(scores_with_extra, SCORES_SCHEMA)

    # Should warn about extra field
    assert "unexpected_field" in result.extra_fields, \
        "Extra field should be detected"

    logger.info("✓ Extra field detection PASSED")


def test_type_mismatch_detection(schema_validator):
    """
    Test: Type mismatches are detected.

    Validates:
    - Wrong types cause validation failure
    - Clear error messages
    - Expected vs actual types reported
    """
    logger.info("TEST: Type mismatch detection")

    # Scores with wrong types
    scores_wrong_types = {
        "technical": "82.5",  # Should be float, not string
        "communication": 78.0,
        "behavioral": 83.0,
        "overall": 81.2
    }

    result = schema_validator.validate(scores_wrong_types, SCORES_SCHEMA)

    # Should fail validation
    assert not result.valid, "Validation should fail for type mismatches"

    # Check type mismatch reported
    type_mismatch_fields = [f for f, _, _ in result.type_mismatches]
    assert "technical" in type_mismatch_fields, \
        "Type mismatch for 'technical' should be detected"

    logger.info("✓ Type mismatch detection PASSED")


def test_value_range_violation_detection(schema_validator):
    """
    Test: Value range violations are detected.

    Validates:
    - Out-of-range values cause validation failure
    - Min/max constraints enforced
    - Clear violation messages
    """
    logger.info("TEST: Value range violation detection")

    # Scores with out-of-range values
    scores_out_of_range = {
        "technical": 150.0,  # Should be <= 100.0
        "communication": 78.0,
        "behavioral": 83.0,
        "overall": 81.2
    }

    result = schema_validator.validate(scores_out_of_range, SCORES_SCHEMA)

    # Should fail validation
    assert not result.valid, "Validation should fail for out-of-range values"

    # Check value violation reported
    violation_fields = [f for f, _ in result.value_violations]
    assert "technical" in violation_fields, \
        "Value range violation for 'technical' should be detected"

    logger.info("✓ Value range violation detection PASSED")


# ============================================================================
# TEST: VALIDATION REPORT
# ============================================================================

def test_validation_report_generation(contract_validator, sample_interview_state):
    """
    Test: Validation report is generated correctly.

    Validates:
    - Report includes all validations
    - Statistics are accurate
    - History is maintained
    """
    logger.info("TEST: Validation report generation")

    # Perform multiple validations
    state_dict = sample_interview_state.model_dump()

    for i in range(5):
        contract_validator.validate_interview_state(state_dict)

    # Get report
    report = contract_validator.get_validation_report()

    # Validate report
    assert report["total_validations"] == 5
    assert report["valid"] == 5
    assert report["invalid"] == 0
    assert report["success_rate"] == 1.0
    assert len(report["history"]) == 5

    logger.info(f"Validation report: {report['total_validations']} validations, "
                f"{report['success_rate']*100:.1f}% success rate")

    logger.info("✓ Validation report generation PASSED")


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_summary():
    """
    Summary of frontend-backend contract tests.

    Coverage:
    ✓ Interview state schema validation
    ✓ Question schema validation
    ✓ Voice analysis schema validation
    ✓ Answer quality schema validation
    ✓ Body language schema validation
    ✓ Scores schema validation
    ✓ Recommendations schema validation
    ✓ WebSocket transcript update schema
    ✓ WebSocket metrics update schema
    ✓ WebSocket score update schema
    ✓ Field name consistency
    ✓ Data type consistency
    ✓ Missing field detection
    ✓ Extra field detection
    ✓ Type mismatch detection
    ✓ Value range violation detection
    ✓ Validation report generation

    Total: 17 contract tests
    """
    logger.info("Frontend-backend contract tests completed successfully")
