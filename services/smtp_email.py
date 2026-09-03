import asyncio
import smtplib
import os


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sender_email = os.environ.get("GMAIL_EMAIL")
password = os.environ.get("GMAIL_APP_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def _send_email_sync(
    to_email: str,
    subject: str,
    html: str,
):
    msg = MIMEMultipart("alternative")

    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()

        server.login(
            sender_email,
            password,
        )

        server.sendmail(
            sender_email,
            to_email,
            msg.as_string(),
        )

        server.quit()


async def send_email(
    to_email: str,
    subject: str,
    html: str,
):
    await asyncio.to_thread(
        _send_email_sync,
        to_email,
        subject,
        html,
    )