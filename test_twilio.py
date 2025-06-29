from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MY_PHONE_NUMBER = os.getenv("MY_PHONE_NUMBER")

client = Client(account_sid, auth_token)

message = client.messages.create(
    from_='whatsapp:TWILIO_PHONE_NUMBER',  # Twilio sandbox WhatsApp number
    body='🌾 Hello from Field Doctor! Your weather alert system is working.',
    to='whatsapp:MY_PHONE_NUMBER'  # Replace with your verified number
)

print(message.sid)
