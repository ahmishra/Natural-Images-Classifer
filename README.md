# Natural-Images-Classifer
Classifies between natural images, given an image.


## How to run
It requires Python 3.8+ and a CUDA enabled GPU.<br>
To run the file, it is first reccomended to create a virtual environment either by using `conda` or `pipenv`.

1. Clone the repository
2. Switch to the directory where you cloned it
3. Open your Terminal or Command Line and enter this command `pip install -r requirements.txt` to install the python requirements for this project
4. Once succesfully installed run the script by doing `python predcit.py` or `python3 predict.py` 
5. Provide the link to a image which is either of these categories: `'airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person'`

The model will then process the image and predict what it is
