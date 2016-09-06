from data import make_dataset, save_submission
from features import build_features
from models import train_model, predict_model

def main():
	train, test, words, users = make_dataset.main()
	X, y, final_test = build_features.main(train, test, words, users)

	pipeline = train_model.fit(X, y)

	submissions = predict_model.predict(pipeline, final_test)

	save_submission.save(submissions)


if __name__ == '__main__':
	main()