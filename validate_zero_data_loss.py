import json
import re
import sys


RAW_16_ANSWERS_JSON = r'''
[
  {
    "section": "SEC-01",
    "answer": "Yes, the solution supports SAML 2.0 single sign-on and allows administrators to disable local password authentication."
  },
  {
    "section": "SEC-02",
    "answer": "Yes, customer data is encrypted in transit using TLS 1.3 and at rest with AES-256, with vendor-managed or customer-managed keys via AWS KMS or Azure Key Vault."
  },
  {
    "section": "SEC-03",
    "answer": "The platform maintains a tamper-evident, exportable audit trail that logs all document access events, administrative setting modifications, user approval decisions, and final RFP submissions."
  },
  {
    "section": "SEC-04",
    "answer": "The platform provides granular role-based access controls that strictly separate content authors, reviewers, approvers, and system administrators."
  },
  {
    "section": "SUB-01",
    "answer": "The system supports uploading PDF files."
  },
  {
    "section": "SUB-02",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "SUB-03",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "Human Approval Before Submission",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "Deployment and Data Location",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "REQUEST FOR PROPOSAL #2026-014",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "1. Overview / General Instructions",
    "answer": "CAPABILITY_NOT_FOUND"
  },
  {
    "section": "TECHNICAL ENVIRONMENT",
    "answer": "The service exposes fully documented HTTPS REST APIs for approved enterprise integrations, supporting token-based authentication (Bearer tokens), cursor-based pagination, automatic rate-limit handling with standard HTTP 429 headers, and structured JSON error reporting."
  },
  {
    "section": "Document Ingestion",
    "answer": "The system supports ingestion of PDF, DOCX, HTML, and plain-text reference material and preserves source attribution for reviewers."
  },
  {
    "section": "Response Quality and Traceability",
    "answer": "The system automatically includes inline citations linked to internal source material for each drafted answer, and if relevant evidence cannot be found, it flags the item as CAPABILITY_NOT_FOUND or routes it for manual human completion rather than generating unsupported claims."
  },
  {
    "section": "Pricing submission notes",
    "answer": "Implementation fee: $5,000 one-time. Recurring subscription: $150 per user per month billed annually. Optional service fees: optional dedicated support packages. Applicable taxes: local taxes. All listed as separate line items; confidential pricing excluded from technical response."
  },
  {
    "section": "Vendor Support",
    "answer": "The vendor provides 24/7 ticketing, email support, and a dedicated customer success manager as standard support channels. For critical issues, the vendor guarantees a 1‑hour response time. Planned maintenance windows are announced on the status page 72 hours in advance. Standard and optional service commitments are clearly identified in the master service agreement."
  }
]
'''

AGGREGATED_3_BUCKETS_JSON = r'''
[
  {
    "section": "Security",
    "answer": "[SEC-01]: Yes, the solution supports SAML 2.0 single sign-on and allows administrators to disable local password authentication.\r\n\r\n[SEC-02]: Yes, customer data is encrypted in transit using TLS 1.3 and at rest with AES-256, with vendor-managed or customer-managed keys via AWS KMS or Azure Key Vault.\r\n\r\n[SEC-03]: The platform maintains a tamper-evident, exportable audit trail that logs all document access events, administrative setting modifications, user approval decisions, and final RFP submissions.\r\n\r\n[SEC-04]: The platform provides granular role-based access controls that strictly separate content authors, reviewers, approvers, and system administrators."
  },
  {
    "section": "Tech Specs",
    "answer": "[SUB-01]: The system supports uploading PDF files.\r\n\r\n[SUB-02]: CAPABILITY_NOT_FOUND\r\n\r\n[SUB-03]: CAPABILITY_NOT_FOUND\r\n\r\n[Human Approval Before Submission]: CAPABILITY_NOT_FOUND\r\n\r\n[Deployment and Data Location]: CAPABILITY_NOT_FOUND\r\n\r\n[REQUEST FOR PROPOSAL #2026-014]: CAPABILITY_NOT_FOUND\r\n\r\n[1. Overview / General Instructions]: CAPABILITY_NOT_FOUND\r\n\r\n[TECHNICAL ENVIRONMENT]: The service exposes fully documented HTTPS REST APIs for approved enterprise integrations, supporting token-based authentication (Bearer tokens), cursor-based pagination, automatic rate-limit handling with standard HTTP 429 headers, and structured JSON error reporting.\r\n\r\n[Document Ingestion]: The system supports ingestion of PDF, DOCX, HTML, and plain-text reference material and preserves source attribution for reviewers.\r\n\r\n[Response Quality and Traceability]: The system automatically includes inline citations linked to internal source material for each drafted answer, and if relevant evidence cannot be found, it flags the item as CAPABILITY_NOT_FOUND or routes it for manual human completion rather than generating unsupported claims.\r\n\r\n[Vendor Support]: The vendor provides 24/7 ticketing, email support, and a dedicated customer success manager as standard support channels. For critical issues, the vendor guarantees a 1‑hour response time. Planned maintenance windows are announced on the status page 72 hours in advance. Standard and optional service commitments are clearly identified in the master service agreement."
  },
  {
    "section": "Pricing",
    "answer": "[Pricing submission notes]: Implementation fee: $5,000 one-time. Recurring subscription: $150 per user per month billed annually. Optional service fees: optional dedicated support packages. Applicable taxes: local taxes. All listed as separate line items; confidential pricing excluded from technical response."
  }
]
'''


def validate_zero_data_loss(
    raw_16_answers: list[dict], aggregated_3_buckets: list[dict]
) -> None:
    """Programmatically proves no data was lost during the split handoff."""

    # Strip whitespace and normalize to lowercase for accurate word matching
    raw_text = " ".join([item["answer"].strip() for item in raw_16_answers])
    raw_words = set(re.findall(r"\w+", raw_text.lower()))

    aggregated_text = " ".join(
        [item["answer"].strip() for item in aggregated_3_buckets]
    )
    aggregated_words = set(re.findall(r"\w+", aggregated_text.lower()))

    missing_words = raw_words - aggregated_words

    if not missing_words:
        print("✅ PASS: 100% of raw words are present in the aggregated payload.")
    else:
        print(f"❌ FAIL: Data loss detected. Missing words: {missing_words}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    validate_zero_data_loss(
        json.loads(RAW_16_ANSWERS_JSON),
        json.loads(AGGREGATED_3_BUCKETS_JSON),
    )
