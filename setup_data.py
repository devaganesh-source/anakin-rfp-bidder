from pathlib import Path

def create_sample_files():
    data_dir = Path("sample-data")
    data_dir.mkdir(exist_ok=True)
    
    files = {
        "security.txt": """# Security, Compliance, and Governance Whitepaper

## Single Sign-On and Authentication (SEC-01)
Our platform fully supports single sign-on (SSO) using SAML 2.0 and OIDC integrations with major identity providers (Okta, Azure AD, Google Workspace). Administrators can configure policies to completely disable local password authentication in favor of enforced enterprise SSO.

## Data Encryption Standards (SEC-02)
Customer data is strictly encrypted in transit using industry-standard TLS 1.3 protocols. All data at rest is encrypted using AES-256 encryption. Enterprises can choose between standard vendor-managed keys or bring their own keys (BYOK) via AWS KMS or Azure Key Vault.

## Audit Trails and Compliance (SEC-03)
The platform maintains a tamper-evident, exportable audit trail. It logs all document access events, administrative setting modifications, user approval decisions, and final RFP submissions with precise timestamps and user identifiers.

## Role-Based Access Control (SEC-04)
Granular role-based access controls (RBAC) strictly separate responsibilities between content authors, reviewers, approvers, and system administrators. Custom permission matrices can be defined per workspace.""",

        "architecture.txt": """# Technical Architecture, APIs, and Ingestion

## Technical Environment & APIs (TECHNICAL ENVIRONMENT)
The proposed service exposes fully documented HTTPS REST APIs for approved enterprise integrations. Our API gateway supports token-based authentication (Bearer tokens), cursor-based pagination, automatic rate-limit handling with standard HTTP 429 headers, and structured JSON error reporting.

## Document Ingestion & Source Attribution (Document Ingestion)
Users can seamlessly ingest PDF, DOCX, HTML, and plain-text reference material. The system preserves precise source attribution, allowing reviewers to trace every generated response back to its original reference document snippet.

## Response Quality and Traceability (Response Quality and Traceability)
Every drafted answer automatically includes inline citations linked to internal source material. If relevant factual evidence cannot be found within the vector index, the system automatically flags the item as CAPABILITY_NOT_FOUND or routes it for manual human completion rather than generating unsupported claims.""",

        "pricing.txt": """# Commercial Terms, Pricing, and Vendor Support

## Pricing and Implementation Fees (Pricing submission notes)
- Recurring Subscription: Enterprise licenses are billed at $150 per user per month, billed annually.
- Implementation Fee: A one-time professional services and onboarding implementation fee of $5,000 applies.
- Taxes & Add-ons: Applicable local taxes and optional dedicated support packages are itemized as separate line items on all commercial invoices. Confidential pricing is kept strictly separate from technical response attachments.

## Vendor Support and Maintenance (Vendor Support)
- Standard Support Channels: 24/7 ticketing, email support, and a dedicated customer success manager.
- Escalation Procedures: Critical issues carry a guaranteed 1-hour response time SLA. 
- Maintenance Communications: Planned maintenance windows are communicated via our status page 72 hours in advance. Standard and optional service commitments are clearly designated in the master service agreement."""
    }

    for filename, content in files.items():
        file_path = data_dir / filename
        file_path.write_text(content, encoding="utf-8")
        print(f"Created/Updated: {file_path}")

if __name__ == "__main__":
    create_sample_files()