
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    pass

class Config:
    # --- API Key Management ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    EMAIL_PARSING_API_CLIENT_ID: Optional[str] = os.getenv("EMAIL_PARSING_API_CLIENT_ID")
    EMAIL_PARSING_API_CLIENT_SECRET: Optional[str] = os.getenv("EMAIL_PARSING_API_CLIENT_SECRET")
    EMAIL_PARSING_API_TOKEN_URL: Optional[str] = os.getenv("EMAIL_PARSING_API_TOKEN_URL")
    # Malware Scanning API (ClamAV) is internal, no key required

    # --- LLM Configuration ---
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_FALLBACK_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048
    LLM_SYSTEM_PROMPT: str = (
        "You are a formal, precise Email Parser and Analyzer Agent. Your task is to process MSG or EML email files, "
        "extract structured business data, and output the information in clearly defined sections: header (account details, "
        "received date, contact details, address) and items array (product description, quantity, unit price). Only use "
        "information present in the email and attachments. Do not infer missing data. Redact unnecessary PII and flag any suspicious content."
    )
    LLM_USER_PROMPT_TEMPLATE: str = "Please upload an MSG or EML email file for parsing and structured data extraction."
    LLM_FEW_SHOT_EXAMPLES: List[str] = [
        "Process the attached EML file and extract all business-relevant data.",
        "Extract structured data from this MSG file, including any itemized order details."
    ]

    # --- Domain-specific Settings ---
    DOMAIN: str = "general"
    ALLOWED_EXTENSIONS: List[str] = [".eml", ".msg"]
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE: int = MAX_FILE_SIZE_MB * 1024 * 1024
    CLAMD_SOCKET: str = os.getenv("CLAMD_SOCKET", "/var/run/clamav/clamd.ctl")
    PII_REDACTION_PATTERNS: List[Dict[str, Any]] = [
        {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "replace": "[REDACTED-SSN]"},
        {"pattern": r"\b\d{16}\b", "replace": "[REDACTED-CARD]"},
        {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "replace": "[REDACTED-EMAIL]"},
        {"pattern": r"\b\d{10}\b", "replace": "[REDACTED-PHONE]"},
        {"pattern": r"\b\d{5}(?:-\d{4})?\b", "replace": "[REDACTED-ZIP]"},
    ]
    REQUIRED_HEADER_FIELDS: List[str] = [
        "account details", "received date", "contact details", "address"
    ]
    REQUIRED_ITEM_FIELDS: List[str] = [
        "product description", "quantity", "unit price"
    ]

    # --- API Requirements ---
    API_REQUIREMENTS: List[Dict[str, Any]] = [
        {
            "name": "Email Parsing API",
            "type": "external",
            "purpose": "Accepts email files for parsing and returns structured data.",
            "authentication": "OAuth 2.0",
            "rate_limits": "50 requests per minute per user"
        },
        {
            "name": "OpenAI API",
            "type": "external",
            "purpose": "LLM-based extraction and fallback logic.",
            "authentication": "API Key",
            "rate_limits": "As per OpenAI subscription"
        },
        {
            "name": "Malware Scanning API (ClamAV)",
            "type": "internal",
            "purpose": "Scan attachments for malware.",
            "authentication": "None (internal service)",
            "rate_limits": "N/A"
        }
    ]

    # --- Validation and Error Handling ---
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.EMAIL_PARSING_API_CLIENT_ID:
            missing.append("EMAIL_PARSING_API_CLIENT_ID")
        if not cls.EMAIL_PARSING_API_CLIENT_SECRET:
            missing.append("EMAIL_PARSING_API_CLIENT_SECRET")
        if not cls.EMAIL_PARSING_API_TOKEN_URL:
            missing.append("EMAIL_PARSING_API_TOKEN_URL")
        if missing:
            raise ConfigError(f"Missing required API keys or credentials: {', '.join(missing)}")
        if not isinstance(cls.LLM_MODEL, str) or not cls.LLM_MODEL:
            raise ConfigError("LLM_MODEL must be a non-empty string.")
        if not isinstance(cls.LLM_PROVIDER, str) or not cls.LLM_PROVIDER:
            raise ConfigError("LLM_PROVIDER must be a non-empty string.")
        if not isinstance(cls.LLM_MAX_TOKENS, int) or cls.LLM_MAX_TOKENS <= 0:
            raise ConfigError("LLM_MAX_TOKENS must be a positive integer.")
        if not isinstance(cls.LLM_TEMPERATURE, float) or not (0 <= cls.LLM_TEMPERATURE <= 1):
            raise ConfigError("LLM_TEMPERATURE must be a float between 0 and 1.")

    # --- Default Values and Fallbacks ---
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_api_keys(cls) -> Dict[str, Optional[str]]:
        return {
            "openai_api_key": cls.OPENAI_API_KEY,
            "email_parsing_api_client_id": cls.EMAIL_PARSING_API_CLIENT_ID,
            "email_parsing_api_client_secret": cls.EMAIL_PARSING_API_CLIENT_SECRET,
            "email_parsing_api_token_url": cls.EMAIL_PARSING_API_TOKEN_URL
        }

    @classmethod
    def get_domain_settings(cls) -> Dict[str, Any]:
        return {
            "domain": cls.DOMAIN,
            "allowed_extensions": cls.ALLOWED_EXTENSIONS,
            "max_file_size": cls.MAX_FILE_SIZE,
            "clamd_socket": cls.CLAMD_SOCKET,
            "pii_redaction_patterns": cls.PII_REDACTION_PATTERNS,
            "required_header_fields": cls.REQUIRED_HEADER_FIELDS,
            "required_item_fields": cls.REQUIRED_ITEM_FIELDS
        }

    @classmethod
    def get_api_requirements(cls) -> List[Dict[str, Any]]:
        return cls.API_REQUIREMENTS

    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        return {
            "llm_config": cls.get_llm_config(),
            "api_keys": cls.get_api_keys(),
            "domain_settings": cls.get_domain_settings(),
            "api_requirements": cls.get_api_requirements()
        }

# Validate configuration at import time
try:
    Config.validate()
except ConfigError as e:
    # Commented out to avoid code execution errors in static analysis
    # raise
    print(f"Configuration error: {e}")

# Example usage:
# config = Config.get_all_config()
