from email import policy
from email.parser import BytesParser
from email.header import decode_header
from bs4 import BeautifulSoup

# Decode MIME encoded headers
def decode_mime_header(header_value):
    if header_value is None:
        return ""
    decoded_parts = decode_header(header_value)
    header = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            header += part.decode(encoding or 'utf-8', errors="ignore")
        else:
            header += part
    return header

# Extract sender, receiver, datetime, subject and body
# from the uploaded file
def extract_eml_content_from_upload(uploaded_file):

    # Read the file content
    msg = BytesParser(policy=policy.default).parse(uploaded_file)

    # Extract email headers
    sender = decode_mime_header(msg.get('From'))
    receiver = decode_mime_header(msg.get('To'))
    date = msg.get('Date')
    subject = decode_mime_header(msg.get('Subject'))

    # Extract email body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                body += part.get_content()
            elif content_type == "text/html" and not body:
                html_content = part.get_content()
                body += BeautifulSoup(html_content, "html.parser").get_text()
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            body = msg.get_content()
        elif content_type == "text/html":
            body = BeautifulSoup(msg.get_content(), "html.parser").get_text()

    return {
        "sender": sender,
        "receiver": receiver,
        "datetime": date,
        "subject": subject,
        "body": body.strip()
    }