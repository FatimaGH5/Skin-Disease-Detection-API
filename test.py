import requests

url = 'https://kkkkkk-e6d5.onrender.com/predict'  # Replace this with your Flask app's URL

# Load an image file
files = {'file': open('C:/Users/User/Downloads/archive (5)/HAM1000_images/HAM1000_images/ISIC_0034215.jpg', 'rb')}  # Replace 'path_to_your_image.jpg' with the path to your image file

# Send the POST request
response = requests.post(url, files=files)

# Print the response
print(response.content)

#print(response.json())