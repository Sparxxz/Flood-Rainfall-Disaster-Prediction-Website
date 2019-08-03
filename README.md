# Flood-Rainfall-Disaster-Prediction-Website
# NTT-CodeForGood-DataScience-HACKATHON (Finalist)
## Disaster relief website from flood and rainfall.

## Welcome Page
![1](https://user-images.githubusercontent.com/39629176/55099197-4f51bb00-50e5-11e9-9e8e-d1ca8767c740.jpg)

## Approach:->
Disaster response is the second phase of the disaster Management cycle. It consists of a number of elements, for example, warning, evacuation, search and rescue, providing immediate assistance, assessing damage, continuing assistance and the immediate restoration.
So among all, we have worked upon warning system for floods. In this, we have provided a user interface to the common public to check the level of water flow in rivers in future and have a provided a mechanism of notification if there is any possibility of flood due to any river in nearby future(12 months). Along with that users can also see the historical trends of rivers flow and can visualize the rainfall patterns also in their Sub-Division(Area).
So with that much information beforehand and knowing the chances of the flood in any region we can prepare ourselves and alert the local public so that loss would be minimum.

## Workflow chart
![1](https://user-images.githubusercontent.com/39629176/55099884-912f3100-50e6-11e9-96bb-d66f9538f4b7.jpg)

## STEPS TAKEN IN THE PROCESS:->
#### CONNECTION TO HTML:
1.	 A user issues a request for a domain's root URL / to go to its index page
2.	main.py maps the URL / to a Python function
3.	The Python function finds a web template living in the templates/ folder.
4.	A web template will look in the static/ folder for any images, CSSfiles it needs as it renders to HTML
5.	Rendered HTML is sent back to main.py
6.	main.py sends the HTML back to the browser

#### URL IN THE BROWSER AND BACKEND CONNECTION:
1.	First. we imported the Flask class and a function render template.
2.	Next, we created a new instance of the Flask class.
3.	We then mapped the URL / to the function index(). Now, when someone visits this URL, the function index() will execute.
4.	The function index() uses the Flask function render template() to render the index.html template we just created from the templates/ folder to the browser.
5.	Finally, we use run() to run our app on a local server.
6.	We'll set the debug flag to true, so we can view any applicable error messages if something goes wrong, and so that the local server automatically reloads after we've made changes to the code.
7.	 When we visited http://127.0.0.1:5000/, main.py had code in it, which mapped the URL / to the Python function index().
8.	index() found the web template index.html in the templates/ folder, rendered it to HTML, and sent it back to the browser, giving us the screen above.

