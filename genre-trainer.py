#python class for training genre.

from clarifai import rest
from clarifai.rest import Image as ClImage
from clarifai.rest import ClarifaiApp

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

app = ClarifaiApp(quiet=True)


# app.inputs.create_image_from_filename("/Users/johnsloan/Dropbox/test2/genres-spectrogram/train/blues/blues_0.png", image_id="blues_0", concepts=['blues'])
# app.inputs.delete("blues_0")
def add_inputs_with_class(genre=""):
	for i in range(0,100):
		# avoid every tenth file to save 10 files in each genre for testing
		if (i != 0 and (i+1) % 10 != 0) or i > 98:
			app.inputs.create_image_from_filename("/Users/johnsloan/Dropbox/test2/genres-spectrogram/train/" + genre + "/" + genre + "_" + str(i) + ".png", image_id=genre + "_" + str(i), concepts=[genre])


def add_single_input(file_number, genre):
	app.inputs.create_image_from_filename("/Users/johnsloan/Dropbox/test2/genres-spectrogram/train/" + genre + "/" + genre + "_" + str(file_number) + ".png", image_id=genre + "_" + str(file_number), concepts=[genre])


# Adds all inputs from spectrogram files
def add_all_inputs():
	for genre in GENRES:
		add_inputs_with_class(genre)

# Train the model
# model = app.models.create("genres-not-exclusive", model_name="genres-not-exclusive", concepts=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"], concepts_mutually_exclusive=False, closed_environment=True)
# model = app.models.create("genres-bad-subset", model_name="genres-bad-subset", concepts=["disco","pop","reggae","rock"], concepts_mutually_exclusive=True, closed_environment=True)
# if model :
# 	model.train()

# Predict on the trained model

def predict_on_file(filename=""):
	model = app.models.get("genres")
	image = ClImage(filename=filename)
	outputs = model.predict([image])['outputs']

	prediction_dict = {}
	for output in outputs:
		concepts = output["data"]["concepts"]
		for concept in concepts:
			concept_name = concept["name"]
			concept_value = concept["value"]
			# print(concept_name + ": " + str(concept_value))
			prediction_dict[concept_name] = concept_value

	return prediction_dict

def test_predict_for_genre(genre):
	print("GENRE: " + genre)
	overall_list = []
	for i in range(0,10):
		if i > 0:
			index = (i * 10) - 1
		else:
			index = 0
		print("predict on file: " + genre + "_" + str(index))
		filename = "/Users/johnsloan/Dropbox/test2/genres-spectrogram/train/" + genre + "/" + genre + "_" + str(index) + ".png"
		overall_list.append(predict_on_file(filename))

	return overall_list


def test_predict_all_genres():
	for genre in GENRES:
		print("=======================")
		test_predict_for_genre(genre)

def get_accuracy(genre="",overall_list=[]):
	count = 0
	for prediction in overall_list:
		highest = ("",0)
		for genre_name in prediction.keys():
			value = prediction[genre_name]
			if value > highest[1]:
				highest = (genre_name,value)
		print(highest)
		if highest[0] == genre:
			count+=1
	return float(count)/len(overall_list)
	

# predict_on_file("/Users/johnsloan/Dropbox/test2/genres-spectrogram/train/disco/disco_19.png")
overall_list = test_predict_for_genre("hiphop")
accuracy = get_accuracy("hiphop",overall_list)
print accuracy
# app.inputs.check_status()

# img = app.inputs.get("blues_0")
# print img