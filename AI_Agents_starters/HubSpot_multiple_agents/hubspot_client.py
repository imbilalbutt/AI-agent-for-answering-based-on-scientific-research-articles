import os
from hubspot import HubSpot
from hubspot.crm.contacts import ApiException
from langchain.tools import tool

api_client = HubSpot(access_token=os.getenv("HUBSPOT_ACCESS_TOKEN"))

def get_contact_by_email(email):
    """Retrieve a contact by email address."""
    try:
        contact = api_client.crm.contacts.basic_api.get_by_id(email, id_property="email")
        return contact
    except ApiException as e:
        if e.status == 404:
            return None
        raise

def create_contact(email, firstname, lastname):
    """Create a new contact in HubSpot."""
    properties = {
        "email": email,
        "firstname": firstname,
        "lastname": lastname
    }
    contact = api_client.crm.contacts.basic_api.create(simple_public_object_input_for_create={"properties": properties})
    return contact

def update_contact(contact_id, properties):
    """Update a contact's properties."""
    return api_client.crm.contacts.basic_api.update(
        contact_id=contact_id,
        simple_public_object_input={"properties": properties}
    )