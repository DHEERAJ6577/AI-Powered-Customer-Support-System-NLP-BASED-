import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Choose a Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_solution(ticket_text, category, faq_context=None):
    """
    Generate AI-based solution for a support ticket.
    """
    prompt = f"""
    You are an AI support assistant.
    A customer submitted the following ticket:

    Ticket: "{ticket_text}"
    Category: {category}

    Context/FAQs: {faq_context if faq_context else "No extra FAQs provided"}

    Provide a helpful, professional, and concise solution for the customer.
    """

    response = model.generate_content(prompt)
    return response.text
