# Validation API Reference

The Validation layer provides data validation, schema validation, and compliance checking for medical data and standards.

## Core Validation Components

### FHIR Validation

::: Medical_KG_rev.validation.fhir
    options:
      show_root_heading: true
      members:
        - FHIRValidator
        - FHIRValidationError
        - _CompiledSchema
        - validate_resource
        - validate_bundle
        - validate_parameters

### UCUM Validation

::: Medical_KG_rev.validation.ucum
    options:
      show_root_heading: true
      members:
        - UCUMValidator
        - UCUMError
        - validate_unit
        - convert_unit
        - get_unit_info

## Usage Examples

### FHIR Resource Validation

```python
from Medical_KG_rev.validation.fhir import FHIRValidator

# Initialize FHIR validator
validator = FHIRValidator(
    fhir_version="R5",
    schemas_dir="schemas/fhir/r5"
)

# Validate a Patient resource
patient_data = {
    "resourceType": "Patient",
    "id": "patient-123",
    "identifier": [
        {
            "use": "usual",
            "type": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                        "code": "MR"
                    }
                ]
            },
            "value": "12345"
        }
    ],
    "name": [
        {
            "use": "official",
            "family": "Doe",
            "given": ["John"]
        }
    ],
    "gender": "male",
    "birthDate": "1990-01-01"
}

try:
    result = await validator.validate_resource("Patient", patient_data)
    if result.valid:
        print("Patient resource is valid")
    else:
        print(f"Validation errors: {result.errors}")
except FHIRValidationError as e:
    print(f"Validation failed: {e}")
```

### FHIR Bundle Validation

```python
# Validate a Bundle resource
bundle_data = {
    "resourceType": "Bundle",
    "id": "bundle-123",
    "type": "collection",
    "entry": [
        {
            "resource": patient_data
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-123",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "33747-0",
                            "display": "Glucose [Mass/volume] in Blood"
                        }
                    ]
                },
                "valueQuantity": {
                    "value": 100,
                    "unit": "mg/dL",
                    "system": "http://unitsofmeasure.org",
                    "code": "mg/dL"
                },
                "subject": {
                    "reference": "Patient/patient-123"
                }
            }
        }
    ]
}

result = await validator.validate_bundle(bundle_data)
if result.valid:
    print("Bundle is valid")
else:
    print(f"Bundle validation errors: {result.errors}")
```

### FHIR Parameters Validation

```python
# Validate FHIR Parameters
parameters_data = {
    "resourceType": "Parameters",
    "parameter": [
        {
            "name": "patient",
            "resource": patient_data
        },
        {
            "name": "observation",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "33747-0"
                        }
                    ]
                },
                "valueQuantity": {
                    "value": 100,
                    "unit": "mg/dL"
                }
            }
        }
    ]
}

result = await validator.validate_parameters(parameters_data)
if result.valid:
    print("Parameters are valid")
else:
    print(f"Parameters validation errors: {result.errors}")
```

### UCUM Unit Validation

```python
from Medical_KG_rev.validation.ucum import UCUMValidator

# Initialize UCUM validator
validator = UCUMValidator(
    ucum_file="data/ucum-essence.xml"
)

# Validate medical units
units_to_validate = [
    "mg/dL",      # Milligrams per deciliter
    "mmol/L",     # Millimoles per liter
    "kg",         # Kilograms
    "cm",         # Centimeters
    "mmHg",       # Millimeters of mercury
    "bpm",        # Beats per minute
    "invalid_unit"  # Invalid unit
]

for unit in units_to_validate:
    try:
        result = validator.validate_unit(unit)
        if result.valid:
            print(f"Unit '{unit}' is valid")
            print(f"  Description: {result.description}")
            print(f"  Dimension: {result.dimension}")
        else:
            print(f"Unit '{unit}' is invalid: {result.error}")
    except UCUMError as e:
        print(f"Error validating unit '{unit}': {e}")
```

### UCUM Unit Conversion

```python
# Convert between units
conversions = [
    ("mg/dL", "mmol/L", 100),  # Convert 100 mg/dL to mmol/L
    ("kg", "g", 70),          # Convert 70 kg to grams
    ("cm", "m", 180),         # Convert 180 cm to meters
    ("mmHg", "kPa", 120)      # Convert 120 mmHg to kPa
]

for from_unit, to_unit, value in conversions:
    try:
        result = validator.convert_unit(value, from_unit, to_unit)
        if result.success:
            print(f"{value} {from_unit} = {result.value} {to_unit}")
        else:
            print(f"Conversion failed: {result.error}")
    except UCUMError as e:
        print(f"Error converting {value} {from_unit} to {to_unit}: {e}")
```

### UCUM Unit Information

```python
# Get detailed unit information
units_info = ["mg/dL", "mmol/L", "kg", "cm", "mmHg"]

for unit in units_info:
    try:
        info = validator.get_unit_info(unit)
        print(f"Unit: {unit}")
        print(f"  Description: {info.description}")
        print(f"  Dimension: {info.dimension}")
        print(f"  Base Units: {info.base_units}")
        print(f"  Conversion Factor: {info.conversion_factor}")
        print()
    except UCUMError as e:
        print(f"Error getting info for unit '{unit}': {e}")
```

### Comprehensive Validation Pipeline

```python
from Medical_KG_rev.validation.fhir import FHIRValidator
from Medical_KG_rev.validation.ucum import UCUMValidator

class MedicalDataValidator:
    """Comprehensive medical data validator."""

    def __init__(self):
        self.fhir_validator = FHIRValidator()
        self.ucum_validator = UCUMValidator()

    async def validate_medical_data(self, data: dict) -> dict:
        """Validate medical data comprehensively."""
        results = {
            "fhir_validation": None,
            "ucum_validation": None,
            "overall_valid": True,
            "errors": []
        }

        # Validate FHIR resource
        if "resourceType" in data:
            try:
                fhir_result = await self.fhir_validator.validate_resource(
                    data["resourceType"],
                    data
                )
                results["fhir_validation"] = fhir_result
                if not fhir_result.valid:
                    results["overall_valid"] = False
                    results["errors"].extend(fhir_result.errors)
            except Exception as e:
                results["overall_valid"] = False
                results["errors"].append(f"FHIR validation error: {e}")

        # Validate UCUM units in the data
        units_found = self._extract_units(data)
        for unit in units_found:
            try:
                ucum_result = self.ucum_validator.validate_unit(unit)
                if not ucum_result.valid:
                    results["overall_valid"] = False
                    results["errors"].append(f"Invalid unit '{unit}': {ucum_result.error}")
            except Exception as e:
                results["overall_valid"] = False
                results["errors"].append(f"UCUM validation error for '{unit}': {e}")

        return results

    def _extract_units(self, data: dict) -> list:
        """Extract units from FHIR data."""
        units = []

        def extract_from_value(value):
            if isinstance(value, dict):
                if "unit" in value:
                    units.append(value["unit"])
                if "code" in value:
                    units.append(value["code"])
                for v in value.values():
                    extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)

        extract_from_value(data)
        return list(set(units))

# Usage
validator = MedicalDataValidator()

# Validate comprehensive medical data
medical_data = {
    "resourceType": "Observation",
    "id": "obs-123",
    "status": "final",
    "code": {
        "coding": [
            {
                "system": "http://loinc.org",
                "code": "33747-0",
                "display": "Glucose [Mass/volume] in Blood"
            }
        ]
    },
    "valueQuantity": {
        "value": 100,
        "unit": "mg/dL",
        "system": "http://unitsofmeasure.org",
        "code": "mg/dL"
    }
}

result = await validator.validate_medical_data(medical_data)
if result["overall_valid"]:
    print("Medical data is valid")
else:
    print(f"Validation errors: {result['errors']}")
```

## Configuration

### FHIR Validation Configuration

```python
# FHIR validator configuration
FHIR_CONFIG = {
    "fhir_version": "R5",
    "schemas_dir": "schemas/fhir/r5",
    "strict_mode": True,
    "validate_references": True,
    "validate_profiles": True,
    "cache_schemas": True,
    "schema_cache_ttl": 3600
}

# FHIR schema configuration
FHIR_SCHEMAS = {
    "Patient": {
        "required_fields": ["id", "name", "gender"],
        "optional_fields": ["birthDate", "identifier"],
        "validation_rules": {
            "gender": ["male", "female", "other", "unknown"],
            "birthDate": "date"
        }
    },
    "Observation": {
        "required_fields": ["id", "status", "code"],
        "optional_fields": ["valueQuantity", "subject"],
        "validation_rules": {
            "status": ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"]
        }
    }
}
```

### UCUM Validation Configuration

```python
# UCUM validator configuration
UCUM_CONFIG = {
    "ucum_file": "data/ucum-essence.xml",
    "strict_mode": True,
    "allow_derived_units": True,
    "cache_units": True,
    "unit_cache_ttl": 3600
}

# Common medical units
MEDICAL_UNITS = {
    "glucose": ["mg/dL", "mmol/L"],
    "weight": ["kg", "g", "lb", "oz"],
    "height": ["cm", "m", "in", "ft"],
    "blood_pressure": ["mmHg", "kPa"],
    "heart_rate": ["bpm", "Hz"],
    "temperature": ["°C", "°F", "K"]
}
```

### Environment Variables

- `FHIR_VERSION`: FHIR version to use (R4, R5)
- `FHIR_SCHEMAS_DIR`: Directory containing FHIR schemas
- `FHIR_STRICT_MODE`: Enable strict FHIR validation
- `UCUM_FILE`: Path to UCUM essence file
- `UCUM_STRICT_MODE`: Enable strict UCUM validation
- `VALIDATION_CACHE_TTL`: Cache TTL for validation results

## Error Handling

### Validation Error Hierarchy

```python
# Base validation error
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

# FHIR validation errors
class FHIRValidationError(ValidationError):
    """FHIR-specific validation errors."""
    pass

# UCUM validation errors
class UCUMError(ValidationError):
    """UCUM-specific validation errors."""
    pass

# Schema validation errors
class SchemaValidationError(ValidationError):
    """Schema validation errors."""
    pass
```

### Error Handling Patterns

```python
from Medical_KG_rev.validation.fhir import FHIRValidationError
from Medical_KG_rev.validation.ucum import UCUMError

try:
    # Validate FHIR resource
    result = await validator.validate_resource("Patient", patient_data)

    if not result.valid:
        # Handle validation errors
        for error in result.errors:
            if error.field == "gender":
                # Handle gender validation error
                logger.warning(f"Invalid gender: {error.value}")
            elif error.field == "birthDate":
                # Handle date validation error
                logger.warning(f"Invalid birth date: {error.value}")
            else:
                # Handle other validation errors
                logger.error(f"Validation error: {error}")

        # Return validation result
        return {"valid": False, "errors": result.errors}

    # Validation successful
    return {"valid": True, "errors": []}

except FHIRValidationError as e:
    # Handle FHIR validation errors
    logger.error(f"FHIR validation failed: {e}")
    return {"valid": False, "errors": [str(e)]}
except UCUMError as e:
    # Handle UCUM validation errors
    logger.error(f"UCUM validation failed: {e}")
    return {"valid": False, "errors": [str(e)]}
```

## Performance Considerations

- **Schema Caching**: FHIR schemas are cached to improve performance
- **Unit Caching**: UCUM units are cached to reduce lookup time
- **Batch Validation**: Support for batch validation of multiple resources
- **Async Operations**: All validation operations are asynchronous
- **Lazy Loading**: Schemas and units are loaded on demand

## Monitoring and Observability

- **Validation Metrics**: Track validation success/failure rates
- **Performance Monitoring**: Monitor validation latency and throughput
- **Error Tracking**: Track validation errors and their frequency
- **Distributed Tracing**: OpenTelemetry spans for validation operations
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Health Checks**: Validation service health check endpoints

## Testing

### Mock Validators

```python
from Medical_KG_rev.validation.fhir import FHIRValidator

class MockFHIRValidator(FHIRValidator):
    """Mock FHIR validator for testing."""

    def __init__(self):
        self.validation_results = {}

    async def validate_resource(self, resource_type: str, data: dict) -> dict:
        """Mock resource validation."""
        # Return predefined validation result
        if resource_type in self.validation_results:
            return self.validation_results[resource_type]

        # Default to valid
        return {"valid": True, "errors": []}

    def set_validation_result(self, resource_type: str, result: dict):
        """Set validation result for testing."""
        self.validation_results[resource_type] = result
```

### Integration Tests

```python
import pytest
from Medical_KG_rev.validation.fhir import FHIRValidator

@pytest.mark.asyncio
async def test_fhir_validation():
    """Test FHIR validation functionality."""
    validator = FHIRValidator()

    # Test valid Patient resource
    patient_data = {
        "resourceType": "Patient",
        "id": "patient-123",
        "name": [{"family": "Doe", "given": ["John"]}],
        "gender": "male"
    }

    result = await validator.validate_resource("Patient", patient_data)
    assert result["valid"] == True
    assert len(result["errors"]) == 0

    # Test invalid Patient resource
    invalid_patient = {
        "resourceType": "Patient",
        "id": "patient-123",
        "gender": "invalid_gender"
    }

    result = await validator.validate_resource("Patient", invalid_patient)
    assert result["valid"] == False
    assert len(result["errors"]) > 0
```
