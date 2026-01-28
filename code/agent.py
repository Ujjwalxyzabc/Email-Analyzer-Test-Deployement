
import os
import re
import asyncio
import tempfile
import shutil
import mimetypes
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv
from loguru import logger
import openai
import extract_msg
import eml_parser
import clamd
from cryptography.fernet import Fernet
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import openpyxl

# =========================
# Configuration Management
# =========================

class Config:
    """Configuration loader and validator."""
    load_dotenv()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    CLAMD_SOCKET: str = os.getenv("CLAMD_SOCKET", "/var/run/clamav/clamd.ctl")
    ALLOWED_EXTENSIONS: Tuple[str, ...] = (".eml", ".msg")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_TEXT_LENGTH: int = 50000
    OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="token")
    LLM_MODEL: str = "gpt-4o"
    LLM_FALLBACK_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048
    SYSTEM_PROMPT: str = (
        "You are a formal, precise Email Parser and Analyzer Agent. Your task is to process MSG or EML email files, "
        "extract structured business data, and output the information in clearly defined sections: header (account details, "
        "received date, contact details, address) and items array (product description, quantity, unit price). Only use "
        "information present in the email and attachments. Do not infer missing data. Redact unnecessary PII and flag any suspicious content."
    )
    USER_PROMPT_TEMPLATE: str = "Please upload an MSG or EML email file for parsing and structured data extraction."
    FEW_SHOT_EXAMPLES: List[str] = [
        "Process the attached EML file and extract all business-relevant data.",
        "Extract structured data from this MSG file, including any itemized order details."
    ]

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        if not cls.ENCRYPTION_KEY:
            raise ValueError("ENCRYPTION_KEY is not set in environment variables.")
        if not isinstance(cls.LLM_MODEL, str) or not cls.LLM_MODEL:
            raise ValueError("LLM_MODEL must be a non-empty string.")

Config.validate()

# =========================
# Logging Configuration
# =========================

logger.add("agent_audit.log", rotation="10 MB", retention="90 days", level="INFO", enqueue=True, backtrace=True, diagnose=True)

# =========================
# Pydantic Models
# =========================

class EmailParseRequest(BaseModel):
    """Request model for email parsing."""
    file: Optional[UploadFile] = None

class EmailParseResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fixing_tips: Optional[str] = None

class ParsedEmail(BaseModel):
    headers: Dict[str, Any]
    body: str
    attachments: List[Dict[str, Any]]

class AttachmentContent(BaseModel):
    filename: str
    content_type: str
    text_content: Optional[str] = None
    malware_detected: bool = False
    malware_info: Optional[str] = None

class ExtractedData(BaseModel):
    header: Dict[str, Any]
    items: List[Dict[str, Any]]
    flagged: Optional[List[str]] = None

# =========================
# Infrastructure Layer
# =========================

class LLMClient:
    """Adapter for OpenAI LLM API with fallback logic."""
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def call_llm(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call LLM with prompt and handle errors/fallback."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed on model {model}: {e}")
            if model != Config.LLM_FALLBACK_MODEL:
                # Fallback to secondary model
                try:
                    response = await self.client.chat.completions.create(
                        model=Config.LLM_FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": Config.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    logger.error(f"LLM fallback call failed: {e2}")
                    raise RuntimeError(f"LLM call failed: {e2}")
            raise RuntimeError(f"LLM call failed: {e}")

# =========================
# Security Service
# =========================

class SecurityService:
    """Handles PII redaction and encryption."""
    PII_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
        (re.compile(r"\b\d{16}\b"), "[REDACTED-CARD]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED-EMAIL]"),
        (re.compile(r"\b\d{10}\b"), "[REDACTED-PHONE]"),
        (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[REDACTED-ZIP]"),
    ]

    def __init__(self, encryption_key: str):
        self.fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)

    def redact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from data."""
        def redact_text(text: str) -> str:
            for pattern, repl in self.PII_PATTERNS:
                text = pattern.sub(repl, text)
            return text

        def recursive_redact(obj):
            if isinstance(obj, dict):
                return {k: recursive_redact(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_redact(i) for i in obj]
            elif isinstance(obj, str):
                return redact_text(obj)
            else:
                return obj

        return recursive_redact(data)

    def encrypt(self, data: str) -> str:
        """Encrypt data using Fernet."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt data using Fernet."""
        return self.fernet.decrypt(token.encode()).decode()

# =========================
# Audit Logger
# =========================

class AuditLogger:
    """Logs events for audit and monitoring."""
    @staticmethod
    def log_event(event_type: str, details: Dict[str, Any]) -> None:
        try:
            logger.info(f"Event: {event_type} | Details: {details}")
        except Exception as e:
            logger.warning(f"Logging failed: {e}")

# =========================
# Error Handler
# =========================

class ErrorHandler:
    """Centralized error handling, retry, escalation."""
    MAX_RETRIES = 3

    @staticmethod
    async def handle_error(error: Exception, context: str) -> Dict[str, Any]:
        logger.error(f"Error in {context}: {error}")
        return {
            "success": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "fixing_tips": ErrorHandler.get_fixing_tips(error)
        }

    @staticmethod
    def get_fixing_tips(error: Exception) -> str:
        if isinstance(error, ValidationError):
            return "Check your input for missing or malformed fields. Ensure JSON is properly formatted."
        elif isinstance(error, HTTPException):
            return "Check your request headers and authentication. Ensure file is valid and within size limits."
        elif isinstance(error, UnicodeDecodeError):
            return "Ensure the uploaded file is a valid text-based email file (EML or MSG)."
        elif isinstance(error, RuntimeError):
            return "Try again later. If the problem persists, contact support."
        else:
            return "Check your input and try again. If the issue persists, contact support."

# =========================
# Domain Layer
# =========================

class EmailFileParser:
    """Parses MSG/EML files and extracts headers, body, attachments."""
    def parse_email(self, file_path: str) -> ParsedEmail:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".eml":
            return self._parse_eml(file_path)
        elif ext == ".msg":
            return self._parse_msg(file_path)
        else:
            raise ValueError("Unsupported file type. Only .eml and .msg are supported.")

    def _parse_eml(self, file_path: str) -> ParsedEmail:
        with open(file_path, 'rb') as f:
            raw_email = f.read()
        parser = eml_parser.EmlParser(include_attachment_data=True, include_raw_body=True)
        parsed = parser.decode_email_bytes(raw_email)
        headers = parsed.get("header", {})
        body = parsed.get("body", {}).get("content", "")
        attachments = []
        for att in parsed.get("attachment", []):
            attachments.append({
                "filename": att.get("filename"),
                "content_type": att.get("content_type"),
                "payload": att.get("payload"),
            })
        return ParsedEmail(headers=headers, body=body, attachments=attachments)

    def _parse_msg(self, file_path: str) -> ParsedEmail:
        msg = extract_msg.Message(file_path)
        headers = {
            "subject": msg.subject,
            "date": msg.date,
            "from": msg.sender,
            "to": msg.to,
            "cc": msg.cc,
        }
        body = msg.body or ""
        attachments = []
        for att in msg.attachments:
            attachments.append({
                "filename": att.longFilename or att.shortFilename,
                "content_type": mimetypes.guess_type(att.longFilename or att.shortFilename)[0] or "application/octet-stream",
                "payload": att.data,
            })
        return ParsedEmail(headers=headers, body=body, attachments=attachments)

class AttachmentHandler:
    """Scans attachments for malware and extracts text from supported formats."""
    def __init__(self, clamd_socket: str):
        try:
            self.cd = clamd.ClamdUnixSocket(path=clamd_socket)
        except Exception as e:
            logger.warning(f"ClamAV socket not available: {e}")
            self.cd = None

    def process_attachments(self, attachments: List[Dict[str, Any]]) -> List[AttachmentContent]:
        results = []
        for att in attachments:
            filename = att.get("filename", "unknown")
            content_type = att.get("content_type", "application/octet-stream")
            payload = att.get("payload")
            malware_detected = False
            malware_info = None
            text_content = None

            # Malware scan
            if self.cd and payload:
                try:
                    scan_result = self.cd.instream(payload if hasattr(payload, 'read') else tempfile.SpooledTemporaryFile())
                    if scan_result and scan_result.get('stream', [None, None])[0] == 'FOUND':
                        malware_detected = True
                        malware_info = scan_result['stream'][1]
                except Exception as e:
                    logger.warning(f"Malware scan failed for {filename}: {e}")

            # Extract text from supported formats
            if not malware_detected and payload:
                try:
                    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(payload)
                            tf.flush()
                            text_content = extract_pdf_text(tf.name)
                        os.unlink(tf.name)
                    elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"] or filename.lower().endswith(".docx"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
                            tf.write(payload)
                            tf.flush()
                            doc = Document(tf.name)
                            text_content = "\n".join([p.text for p in doc.paragraphs])
                        os.unlink(tf.name)
                    elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"] or filename.lower().endswith(".xlsx"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tf:
                            tf.write(payload)
                            tf.flush()
                            wb = openpyxl.load_workbook(tf.name)
                            text_content = ""
                            for ws in wb.worksheets:
                                for row in ws.iter_rows(values_only=True):
                                    text_content += "\t".join([str(cell) if cell is not None else "" for cell in row]) + "\n"
                        os.unlink(tf.name)
                except Exception as e:
                    logger.warning(f"Attachment text extraction failed for {filename}: {e}")

            results.append(AttachmentContent(
                filename=filename,
                content_type=content_type,
                text_content=text_content,
                malware_detected=malware_detected,
                malware_info=malware_info
            ))
        return results

class BusinessDataExtractor:
    """Extracts structured business data from parsed email content using LLM."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def extract_data(self, parsed_email: ParsedEmail, attachments_text: List[str]) -> Dict[str, Any]:
        # Compose prompt
        prompt = self._compose_prompt(parsed_email, attachments_text)
        try:
            llm_response = await self.llm_client.call_llm(
                prompt=prompt,
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            # Try to parse LLM response as JSON
            try:
                import json
                data = json.loads(llm_response)
            except Exception:
                # If not JSON, return as text in a field
                data = {"raw_llm_output": llm_response, "flagged": ["LLM output not in JSON format"]}
            return data
        except Exception as e:
            logger.error(f"BusinessDataExtractor LLM extraction failed: {e}")
            return {"flagged": [f"LLM extraction failed: {e}"]}

    def _compose_prompt(self, parsed_email: ParsedEmail, attachments_text: List[str]) -> str:
        prompt = (
            "Extract the following structured business data from the provided email content and attachments. "
            "Return the result as a JSON object with 'header' and 'items' fields. "
            "Header should include account details, received date, contact details, address. "
            "Items should be an array of objects with product description, quantity, unit price. "
            "Only use information present in the email and attachments. "
            "Redact unnecessary PII and flag any suspicious content.\n\n"
            "Email Headers:\n"
            f"{parsed_email.headers}\n\n"
            "Email Body:\n"
            f"{parsed_email.body[:Config.MAX_TEXT_LENGTH]}\n\n"
        )
        if attachments_text:
            prompt += "Attachment Texts:\n"
            for idx, att_text in enumerate(attachments_text):
                prompt += f"Attachment {idx+1}:\n{att_text[:Config.MAX_TEXT_LENGTH]}\n\n"
        prompt += "Please output strictly in JSON format."
        return prompt

class ValidationService:
    """Validates completeness and correctness of extracted data."""
    REQUIRED_HEADER_FIELDS = ["account details", "received date", "contact details", "address"]

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        header = data.get("header", {})
        items = data.get("items", [])
        for field in self.REQUIRED_HEADER_FIELDS:
            if field not in header or not header[field]:
                errors.append(f"Missing header field: {field}")
        if not isinstance(items, list):
            errors.append("Items should be an array.")
        else:
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"Item {idx+1} is not a valid object.")
                else:
                    for req in ["product description", "quantity", "unit price"]:
                        if req not in item:
                            errors.append(f"Item {idx+1} missing field: {req}")
        return (len(errors) == 0, errors)

# =========================
# Application Layer
# =========================

class EmailParserAgent:
    """Main orchestrator for email parsing and analysis."""
    def __init__(self):
        self.email_parser = EmailFileParser()
        self.attachment_handler = AttachmentHandler(Config.CLAMD_SOCKET)
        self.llm_client = LLMClient(Config.OPENAI_API_KEY)
        self.business_extractor = BusinessDataExtractor(self.llm_client)
        self.validation_service = ValidationService()
        self.security_service = SecurityService(Config.ENCRYPTION_KEY)
        self.audit_logger = AuditLogger()
        self.error_handler = ErrorHandler()

    async def process_email(self, file_path: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.audit_logger.log_event("email_upload", {"user": user_context.get("user_id", "anonymous"), "file_path": file_path})
            parsed_email = self.email_parser.parse_email(file_path)
            self.audit_logger.log_event("email_parsed", {"headers": parsed_email.headers})

            # Process attachments
            attachments_content = self.attachment_handler.process_attachments(parsed_email.attachments)
            attachments_text = [att.text_content for att in attachments_content if att.text_content and not att.malware_detected]
            for att in attachments_content:
                if att.malware_detected:
                    self.audit_logger.log_event("malware_detected", {"filename": att.filename, "info": att.malware_info})

            # Extract business data
            extracted_data = await self.business_extractor.extract_data(parsed_email, attachments_text)
            self.audit_logger.log_event("data_extracted", {"data": extracted_data})

            # Validate
            is_valid, validation_errors = self.validation_service.validate(extracted_data)
            if not is_valid:
                extracted_data.setdefault("flagged", []).extend(validation_errors)

            # Redact PII
            redacted_data = self.security_service.redact(extracted_data)

            # Final output
            response = {
                "success": True,
                "data": redacted_data,
                "error_type": None,
                "error_message": None,
                "fixing_tips": None
            }
            self.audit_logger.log_event("process_complete", {"user": user_context.get("user_id", "anonymous"), "result": "success"})
            return response
        except Exception as e:
            self.audit_logger.log_event("process_failed", {"user": user_context.get("user_id", "anonymous"), "error": str(e)})
            return await self.error_handler.handle_error(e, "process_email")

# =========================
# Presentation Layer (API)
# =========================

app = FastAPI(
    title="Email Parser and Analyzer Agent",
    description="API for parsing MSG/EML email files and extracting structured business data.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 Authentication (dummy for demonstration)
async def get_current_user(token: str = Depends(Config.OAUTH2_SCHEME)):
    # In production, validate token and return user context
    return {"user_id": "test_user"}

# Exception Handlers

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "error_message": str(exc),
            "fixing_tips": "Check your input for missing or malformed fields. Ensure JSON is properly formatted."
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "error_message": exc.detail,
            "fixing_tips": "Check your request headers and authentication. Ensure file is valid and within size limits."
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "fixing_tips": "Check your input and try again. If the issue persists, contact support."
        }
    )

# JSON Error Handler for malformed JSON
@app.middleware("http")
async def json_error_middleware(request: Request, call_next):
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            await request.json()
        except Exception as e:
            logger.error(f"Malformed JSON: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_type": "MalformedJSON",
                    "error_message": "Malformed JSON in request body.",
                    "fixing_tips": "Ensure your JSON is properly formatted. Common issues: missing quotes, trailing commas, or unescaped characters."
                }
            )
    response = await call_next(request)
    return response

# Input Validation Helper
def validate_file_upload(file: UploadFile) -> Tuple[bool, str]:
    if not file:
        return False, "No file uploaded."
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {ext}. Only .eml and .msg are allowed."
    if file.spool_max_size and file.spool_max_size > Config.MAX_FILE_SIZE:
        return False, f"File size exceeds limit of {Config.MAX_FILE_SIZE // (1024*1024)} MB."
    return True, ""

# =========================
# API Endpoints
# =========================

@app.post("/parse-email", response_model=EmailParseResponse, tags=["Email Parsing API"])
async def parse_email_api(
    file: UploadFile = File(..., description="MSG or EML email file"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    # Input validation
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_type": "InvalidFile",
                "error_message": error_msg,
                "fixing_tips": "Upload a valid .eml or .msg file under 10 MB."
            }
        )
    # Save file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        agent = EmailParserAgent()
        result = await agent.process_email(tmp_path, user)
        # Clean up temp file
        os.unlink(tmp_path)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "fixing_tips": "Ensure the file is a valid .eml or .msg email file and try again."
            }
        )

@app.get("/health", tags=["Health"])
async def health_check():
    return {"success": True, "status": "ok"}

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Email Parser and Analyzer Agent API...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
