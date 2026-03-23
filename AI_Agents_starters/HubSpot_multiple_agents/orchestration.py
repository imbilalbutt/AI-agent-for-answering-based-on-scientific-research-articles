import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from hubspot_client import get_contact, add_email_draft_to_contact

llm = ChatOpenAI(model="gpt-4o-mini")

# Agent 1: Fetcher
fetcher = Agent(
    role="HubSpot Contact Fetcher",
    goal="Fetch contact details from HubSpot by email",
    backstory="...",
    tools=[get_contact],
    llm=llm,
    verbose=True
)

# Agent 2: Writer
writer = Agent(
    role="Personalized Email Writer",
    goal="Write a friendly welcome email for a contact",
    backstory="...",
    llm=llm,
    verbose=True
)

# Agent 3: Saver (uses the new tool)
saver = Agent(
    role="Email Draft Saver",
    goal="Save the generated email draft into HubSpot",
    backstory="...",
    tools=[add_email_draft_to_contact],
    llm=llm,
    verbose=True
)

# Tasks
fetch = Task(
    description="Retrieve contact details for email {email} from HubSpot. Return the contact ID and personal info.",
    expected_output="A summary including contact ID, first name, last name.",
    agent=fetcher
)

write = Task(
    description="Write a personalized welcome email. Use the contact's first name. "
                "Output should include subject and body.",
    expected_output="Subject: 'Welcome, [Name]!'\n\nBody: ...",
    agent=writer,
    context=[fetch]
)

save = Task(
    description="Take the email subject and body and save them to HubSpot for the given contact ID.",
    expected_output="Confirmation that the draft was saved.",
    agent=saver,
    context=[fetch, write]
)

crew = Crew(
    agents=[fetcher, writer, saver],
    tasks=[fetch, write, save],
    verbose=2
)

# Run for a specific email
result = crew.kickoff(inputs={"email": "john@example.com"})
print(result)