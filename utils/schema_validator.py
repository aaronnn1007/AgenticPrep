"""
Schema Validator
================
Production-grade schema validation for frontend-backend contract testing.

Validates:
- API response schemas
- WebSocket message formats
- Redis session state schemas
- Data type consistency
- Required field presence
- Value range constraints

This ensures frontend expectations match backend reality.
"""

from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Expected field types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    field_type: FieldType
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None
    nested_schema: Optional['Schema'] = None
    description: str = ""


@dataclass
class Schema:
    """Complete schema definition."""
    name: str
    fields: List[FieldSchema]
    allow_extra_fields: bool = False

    def get_required_fields(self) -> Set[str]:
        """Get set of required field names."""
        return {f.name for f in self.fields if f.required}

    def get_field(self, name: str) -> Optional[FieldSchema]:
        """Get field schema by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    extra_fields: List[str] = field(default_factory=list)
    type_mismatches: List[Tuple[str, str, str]] = field(
        default_factory=list)  # (field, expected, actual)
    value_violations: List[Tuple[str, str]] = field(
        default_factory=list)  # (field, reason)

    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_fields": self.missing_fields,
            "extra_fields": self.extra_fields,
            "type_mismatches": [
                {"field": f, "expected": e, "actual": a}
                for f, e, a in self.type_mismatches
            ],
            "value_violations": [
                {"field": f, "reason": r}
                for f, r in self.value_violations
            ]
        }


class SchemaValidator:
    """
    Validates data against defined schemas.

    Performs:
    - Required field checking
    - Type validation
    - Range validation
    - Enum validation
    - Nested object validation
    """

    def validate(self, data: Dict[str, Any], schema: Schema) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: Schema definition

        Returns:
            Validation result with detailed errors/warnings
        """
        result = ValidationResult(valid=True)

        # Check required fields
        required_fields = schema.get_required_fields()
        data_fields = set(data.keys())

        missing = required_fields - data_fields
        if missing:
            result.missing_fields = list(missing)
            result.add_error(f"Missing required fields: {missing}")

        # Check extra fields
        expected_fields = {f.name for f in schema.fields}
        extra = data_fields - expected_fields
        if extra and not schema.allow_extra_fields:
            result.extra_fields = list(extra)
            result.add_warning(f"Unexpected extra fields: {extra}")

        # Validate each field
        for field_schema in schema.fields:
            field_name = field_schema.name

            if field_name not in data:
                if field_schema.required:
                    continue  # Already reported as missing
                else:
                    continue  # Optional field not present

            field_value = data[field_name]

            # Validate field
            self._validate_field(field_name, field_value, field_schema, result)

        return result

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        schema: FieldSchema,
        result: ValidationResult
    ):
        """Validate single field."""
        # Type validation
        actual_type = self._get_type(value)

        if actual_type != schema.field_type:
            result.type_mismatches.append(
                (field_name, schema.field_type.value, actual_type.value)
            )
            result.add_error(
                f"Field '{field_name}': expected {schema.field_type.value}, "
                f"got {actual_type.value}"
            )
            return  # Skip further validation if type is wrong

        # Range validation for numbers
        if schema.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            if schema.min_value is not None and value < schema.min_value:
                result.value_violations.append(
                    (field_name, f"Value {value} < minimum {schema.min_value}")
                )
                result.add_error(
                    f"Field '{field_name}': value {value} below minimum {schema.min_value}"
                )

            if schema.max_value is not None and value > schema.max_value:
                result.value_violations.append(
                    (field_name, f"Value {value} > maximum {schema.max_value}")
                )
                result.add_error(
                    f"Field '{field_name}': value {value} above maximum {schema.max_value}"
                )

        # Enum validation
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                result.value_violations.append(
                    (field_name, f"Value '{value}' not in allowed values")
                )
                result.add_error(
                    f"Field '{field_name}': value '{value}' not in allowed values {schema.allowed_values}"
                )

        # Nested object validation
        if schema.field_type == FieldType.OBJECT and schema.nested_schema:
            nested_result = self.validate(value, schema.nested_schema)
            if not nested_result.valid:
                result.valid = False
                result.errors.append(
                    f"Nested validation failed for '{field_name}': {nested_result.errors}"
                )

        # Array validation
        if schema.field_type == FieldType.ARRAY and schema.nested_schema:
            for idx, item in enumerate(value):
                nested_result = self.validate(item, schema.nested_schema)
                if not nested_result.valid:
                    result.valid = False
                    result.errors.append(
                        f"Array item validation failed for '{field_name}[{idx}]': {nested_result.errors}"
                    )

    def _get_type(self, value: Any) -> FieldType:
        """Determine field type from value."""
        if value is None:
            return FieldType.NULL
        elif isinstance(value, bool):
            return FieldType.BOOLEAN
        elif isinstance(value, int):
            return FieldType.INTEGER
        elif isinstance(value, float):
            return FieldType.FLOAT
        elif isinstance(value, str):
            return FieldType.STRING
        elif isinstance(value, list):
            return FieldType.ARRAY
        elif isinstance(value, dict):
            return FieldType.OBJECT
        else:
            return FieldType.STRING  # Default fallback


# ============================================================================
# PREDEFINED SCHEMAS FOR FRONTEND-BACKEND CONTRACT
# ============================================================================

# Question Schema
QUESTION_SCHEMA = Schema(
    name="Question",
    fields=[
        FieldSchema("text", FieldType.STRING, required=True),
        FieldSchema("topic", FieldType.STRING, required=True),
        FieldSchema("difficulty", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("intent", FieldType.ARRAY, required=False),
    ]
)

# Voice Analysis Schema
VOICE_ANALYSIS_SCHEMA = Schema(
    name="VoiceAnalysis",
    fields=[
        FieldSchema("clarity", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=1.0),
        FieldSchema("speech_rate_wpm", FieldType.FLOAT,
                    required=True, min_value=0.0),
        FieldSchema("filler_ratio", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("tone", FieldType.STRING, required=True),
    ]
)

# Answer Quality Schema
ANSWER_QUALITY_SCHEMA = Schema(
    name="AnswerQuality",
    fields=[
        FieldSchema("relevance", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=1.0),
        FieldSchema("correctness", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("depth", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=1.0),
        FieldSchema("structure", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=1.0),
        FieldSchema("gaps", FieldType.ARRAY, required=False),
    ]
)

# Body Language Schema
BODY_LANGUAGE_SCHEMA = Schema(
    name="BodyLanguage",
    fields=[
        FieldSchema("eye_contact", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("posture_stability", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("facial_expressiveness", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("distractions", FieldType.ARRAY, required=False),
    ]
)

# Confidence Behavior Schema
CONFIDENCE_BEHAVIOR_SCHEMA = Schema(
    name="ConfidenceBehavior",
    fields=[
        FieldSchema("confidence", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("nervousness", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("professionalism", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=1.0),
        FieldSchema("behavioral_flags", FieldType.ARRAY, required=False),
    ]
)

# Scores Schema
SCORES_SCHEMA = Schema(
    name="Scores",
    fields=[
        FieldSchema("technical", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=100.0),
        FieldSchema("communication", FieldType.FLOAT,
                    required=True, min_value=0.0, max_value=100.0),
        FieldSchema("behavioral", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=100.0),
        FieldSchema("overall", FieldType.FLOAT, required=True,
                    min_value=0.0, max_value=100.0),
    ]
)

# Recommendations Schema
RECOMMENDATIONS_SCHEMA = Schema(
    name="Recommendations",
    fields=[
        FieldSchema("strengths", FieldType.ARRAY, required=True),
        FieldSchema("weaknesses", FieldType.ARRAY, required=True),
        FieldSchema("improvement_plan", FieldType.ARRAY, required=True),
    ]
)

# Complete Interview State Schema
INTERVIEW_STATE_SCHEMA = Schema(
    name="InterviewState",
    fields=[
        FieldSchema("interview_id", FieldType.STRING, required=True),
        FieldSchema("role", FieldType.STRING, required=True),
        FieldSchema("experience_level", FieldType.STRING, required=True),
        FieldSchema("question", FieldType.OBJECT, required=False,
                    nested_schema=QUESTION_SCHEMA),
        FieldSchema("transcript", FieldType.STRING, required=False),
        FieldSchema("voice_analysis", FieldType.OBJECT,
                    required=False, nested_schema=VOICE_ANALYSIS_SCHEMA),
        FieldSchema("answer_quality", FieldType.OBJECT,
                    required=False, nested_schema=ANSWER_QUALITY_SCHEMA),
        FieldSchema("body_language", FieldType.OBJECT,
                    required=False, nested_schema=BODY_LANGUAGE_SCHEMA),
        FieldSchema("confidence_behavior", FieldType.OBJECT,
                    required=False, nested_schema=CONFIDENCE_BEHAVIOR_SCHEMA),
        FieldSchema("scores", FieldType.OBJECT, required=False,
                    nested_schema=SCORES_SCHEMA),
        FieldSchema("recommendations", FieldType.OBJECT,
                    required=False, nested_schema=RECOMMENDATIONS_SCHEMA),
    ],
    allow_extra_fields=True  # Allow metadata fields
)

# WebSocket Message Schemas
WS_TRANSCRIPT_UPDATE_SCHEMA = Schema(
    name="TranscriptUpdate",
    fields=[
        FieldSchema("type", FieldType.STRING, required=True,
                    allowed_values={"transcript_update"}),
        FieldSchema("session_id", FieldType.STRING, required=True),
        FieldSchema("transcript", FieldType.STRING, required=True),
        FieldSchema("timestamp", FieldType.STRING, required=True),
    ]
)

WS_METRICS_UPDATE_SCHEMA = Schema(
    name="MetricsUpdate",
    fields=[
        FieldSchema("type", FieldType.STRING, required=True,
                    allowed_values={"metrics_update"}),
        FieldSchema("session_id", FieldType.STRING, required=True),
        FieldSchema("metrics", FieldType.OBJECT, required=True),
        FieldSchema("timestamp", FieldType.STRING, required=True),
    ]
)

WS_SCORE_UPDATE_SCHEMA = Schema(
    name="ScoreUpdate",
    fields=[
        FieldSchema("type", FieldType.STRING, required=True,
                    allowed_values={"score_update"}),
        FieldSchema("session_id", FieldType.STRING, required=True),
        FieldSchema("scores", FieldType.OBJECT, required=True,
                    nested_schema=SCORES_SCHEMA),
        FieldSchema("timestamp", FieldType.STRING, required=True),
    ]
)


class ContractValidator:
    """
    High-level contract validator for frontend-backend integration.

    Validates complete API workflows and WebSocket message flows.
    """

    def __init__(self):
        """Initialize contract validator."""
        self.validator = SchemaValidator()
        self.validation_history: List[Dict[str, Any]] = []

    def validate_interview_state(
        self,
        state: Dict[str, Any],
        check_completeness: bool = False
    ) -> ValidationResult:
        """
        Validate interview state object.

        Args:
            state: Interview state dictionary
            check_completeness: If True, ensure all optional fields are present

        Returns:
            Validation result
        """
        result = self.validator.validate(state, INTERVIEW_STATE_SCHEMA)

        # Additional completeness check
        if check_completeness:
            required_for_complete = [
                "question", "transcript", "voice_analysis", "answer_quality",
                "body_language", "confidence_behavior", "scores", "recommendations"
            ]
            missing_complete = [
                f for f in required_for_complete if f not in state or not state[f]]
            if missing_complete:
                result.add_warning(
                    f"Incomplete interview state. Missing: {missing_complete}"
                )

        # Record validation
        self.validation_history.append({
            "timestamp": str(datetime.now()),
            "schema": "InterviewState",
            "result": result.to_dict()
        })

        return result

    def validate_websocket_message(
        self,
        message: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate WebSocket message format.

        Args:
            message: WebSocket message

        Returns:
            Validation result
        """
        # Determine message type
        msg_type = message.get("type")

        schema_map = {
            "transcript_update": WS_TRANSCRIPT_UPDATE_SCHEMA,
            "metrics_update": WS_METRICS_UPDATE_SCHEMA,
            "score_update": WS_SCORE_UPDATE_SCHEMA,
        }

        schema = schema_map.get(msg_type)
        if not schema:
            result = ValidationResult(valid=False)
            result.add_error(f"Unknown message type: {msg_type}")
            return result

        result = self.validator.validate(message, schema)

        # Record validation
        self.validation_history.append({
            "timestamp": str(datetime.now()),
            "schema": f"WebSocket_{msg_type}",
            "result": result.to_dict()
        })

        return result

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get summary report of all validations.

        Returns:
            Validation report with statistics
        """
        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v["result"]["valid"])
        invalid = total - valid

        return {
            "total_validations": total,
            "valid": valid,
            "invalid": invalid,
            "success_rate": valid / total if total > 0 else 0.0,
            "history": self.validation_history
        }
