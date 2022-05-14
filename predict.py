import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from numpy import argmax, max
from secrets import token_hex
from tensorflow.nn import softmax
from tensorflow import expand_dims
from tensorflow import __version__
from skimage.transform import resize
from skimage.io import imread, imshow
from urllib.request import urlretrieve
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from matplotlib.pyplot import show, title, axis
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model("model.h5")
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

print(f"Using: {device_lib.list_local_devices()[-1].physical_device_desc}")
print(f"Tensorflow v{__version__}")


def predict(img_width=180, img_height=180):
	url = input("URL to image: ")

	print("Generating secure token...")
	ses_token = token_hex(6)

	print("Saving...")
	urlretrieve(url, f"test_images/{ses_token}.png")

	print("Loading..")
	img = load_img(
	    f"test_images/{ses_token}.png", target_size=(img_height, img_width)
	)

	print("Processing...")
	img_array = img_to_array(img)

	print("Displaying...")


	imshow(resize(imread(f"test_images/{ses_token}.png"), (180, 180, 3)))
	title("Processed Image")
	axis("off")
	show()


	print("Predicting...")
	img_array = expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)

	print("Scoring...")
	score = softmax(predictions[0])

	print()
	print(
	    "This image most likely belongs to {} with a {:.2f} percent confidence."
	    .format(class_names[argmax(score)], 100 * max(score))
	)


predict()
