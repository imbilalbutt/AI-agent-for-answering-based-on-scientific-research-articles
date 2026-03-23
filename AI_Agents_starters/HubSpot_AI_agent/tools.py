from langchain.tools import tool
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