import requests

url = 'http://127.0.0.1:5000/predict'
image_path = 'leaf1.jpg'  # Replace with the actual image path

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())
