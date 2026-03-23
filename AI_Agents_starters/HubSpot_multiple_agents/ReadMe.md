Let’s create a simple pipeline that:

Triggers when a new contact is added to HubSpot (we’ll simulate by reading a list of recent contacts).

Generates a personalized email draft.

Stores the draft in a HubSpot custom property (or sends via email).

We’ll use a hybrid approach: a Python script that runs periodically (e.g., via cron) and uses CrewAI for the reasoning steps.