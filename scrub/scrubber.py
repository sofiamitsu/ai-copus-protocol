from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "US_SSN", "CREDIT_CARD"]

_analyzer = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def scrub_transcript(text: str) -> str:
    """
    Removes PII (names, emails, phone numbers, etc.) from transcript text.
    Returns scrubbed text with PII replaced by [REDACTED].
    """
    if not text:
        return text

    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(text=text, entities=_ENTITIES, language="en")
    operators = {entity: OperatorConfig("replace", {"new_value": "[REDACTED]"}) for entity in _ENTITIES}
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text
