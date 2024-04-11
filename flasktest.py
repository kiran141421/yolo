import requests
import base64
from io import BytesIO

with open("images/people.jpg", 'rb') as file:
    file_contents = file.read()

encoded_content = base64.b64encode(file_contents)

encoded_string = encoded_content.decode('utf-8')

data = {
    'file_data': encoded_string
}

# response = requests.get("http://127.0.0.1/")

response = requests.post("http://127.0.0.1:5000/image", json=data)
print(response.text)