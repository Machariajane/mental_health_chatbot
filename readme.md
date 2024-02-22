create a data directory that contains your files (.pdf)

Specify model path

create a .env file and inside specify your 

username = "windows username"
password = "windows password"

run python main.py

On the  browser :
for swaggerui http://localhost:8090/docs
for redoc http://localhost:8090/redoc

for postman
Set the request method to POST.
Enter the URL http://localhost:8090/chat.
Go to the "Body" tab, select "raw", and choose JSON from the dropdown.
Enter the request body in JSON format, e.g., {"message": "Your message here"}.
Hit "Send" and view the response.



