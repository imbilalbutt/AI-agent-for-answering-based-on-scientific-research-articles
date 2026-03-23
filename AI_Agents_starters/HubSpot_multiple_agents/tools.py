from hubspot_client import *

@tool
def get_contact(email: str) -> str:
    """Retrieve contact information from HubSpot by email address."""
    contact = get_contact_by_email(email)
    if contact:
        props = contact.properties
        return f"Contact found: {props.get('firstname')} {props.get('lastname')} ({props.get('email')})"
    else:
        return f"No contact found with email {email}."

@tool
def add_contact(email: str, firstname: str, lastname: str) -> str:
    """Add a new contact to HubSpot with given email, firstname, lastname."""
    contact = create_contact(email, firstname, lastname)
    return f"Contact created with ID: {contact.id}"

@tool
def add_email_draft_to_contact(contact_id: str, subject: str, body: str) -> str:
    """Stores an email draft (subject and body) in HubSpot custom properties."""
    properties = {
        "email_draft_subject": subject,
        "email_draft_body": body
    }
    update_contact(contact_id, properties)
    return f"Email draft saved for contact {contact_id}."