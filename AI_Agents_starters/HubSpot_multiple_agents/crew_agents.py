import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from hubspot_client import get_contact_by_email  # our earlier functions

llm = ChatOpenAI(model="gpt-4o-mini")

# Agent 1: Contact Fetcher (uses the HubSpot tool)
contact_fetcher = Agent(
    role="HubSpot Contact Retriever",
    goal="Fetch contact details from HubSpot by email",
    backstory="You are a CRM specialist who can access HubSpot and retrieve contact information.",
    tools=[get_contact],   # we need to wrap get_contact as a tool for CrewAI
    llm=llm,
    verbose=True
)

# Agent 2: Email Writer
email_writer = Agent(
    role="Personalized Email Writer",
    goal="Write a friendly, personalized welcome email for a new contact",
    backstory="You are a copywriter who crafts engaging emails based on contact details.",
    llm=llm,
    verbose=True
)

# Tasks
fetch_contact_task = Task(
    description="Retrieve contact details for email {email} from HubSpot. Return the full contact information.",
    expected_output="A summary of the contact's first name, last name, and email.",
    agent=contact_fetcher
)

write_email_task = Task(
    description="Using the contact information obtained, write a short welcome email. "
                "The email should be friendly and mention the contact's name. "
                "Subject: 'Welcome, [First Name]!'",
    expected_output="The email content (subject and body).",
    agent=email_writer,
    context=[fetch_contact_task]   # depends on previous task
)

# Form the crew
crew = Crew(
    agents=[contact_fetcher, email_writer],
    tasks=[fetch_contact_task, write_email_task],
    verbose=2
)

# Run the crew for a specific email
result = crew.kickoff(inputs={"email": "john@example.com"})
print(result)