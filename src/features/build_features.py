import pandas as pd
from data import make_dataset


def merge_dataframes(train, test, words, users):
	"""
	Creates training and test dataframes
	by combining information from users and words dataframes
	"""

	train_merged = pd.merge(train, words, left_on=['Artist', 'User'], right_on=['Artist', 'User'], how='left')
	train_merged = pd.merge(train_merged, users, left_on=['User'], right_on=['RESPID'], how='left')

	test_merged = pd.merge(test, words, left_on=['Artist', 'User'], right_on=['Artist', 'User'], how='left')
	test_merged = pd.merge(test_merged, users, left_on=['User'], right_on=['RESPID'], how='left')

	return train_merged, test_merged


def generate_features(train, train_merged, test_merged):
	"""
	Features Based on the user.
		Gender
		Age
		Music
		REGION
		LIST_OWN
		LIST_BACK
		Response to different questions

	Features Based on the artist.
		Heard_Of
		Own_artist_music
		Like Artist
		Characteristics of the songs sung by the artist ( Edgy, Uninspired etc. )

	Features Based on the pair of user and artist.
		Mean Artist Rating
		Min Artist Rating
		Max Artist Rating
		Median Artist Rating
		Mean User Rating
		Min User Rating
		Max User Rating
		Median User Rating
		Mean Rating given to Artist by a User
		Min Rating given to Artist by a User
		Max Rating given to Artist by a User
		Median Rating given to Artist by a User
	"""

	# group by user
	user_group = train.groupby('User')

	# group by artist
	artist_group = train.groupby('Artist')

	# mean user rating based on the training set
	mean_user_ratings = user_group['Rating'].mean().to_dict()

	# min user rating based on the training set
	min_user_ratings = user_group['Rating'].min().to_dict()

	# max user rating based on the training set
	max_user_ratings = user_group['Rating'].max().to_dict()

	# median user rating based on the training set
	median_user_ratings = user_group['Rating'].median().to_dict()


	# mean artist rating based on the training set
	mean_artist_ratings = artist_group['Rating'].mean().to_dict()

	# min artist rating based on the training set
	min_artist_ratings = artist_group['Rating'].min().to_dict()

	# max artist rating based on the training set
	max_artist_ratings = artist_group['Rating'].max().to_dict()

	# median artist rating based on the training set
	median_artist_ratings = artist_group['Rating'].median().to_dict()

	train_merged['mean_user_rating'] = train_merged.User.map(lambda x: mean_user_ratings[x] if x in mean_user_ratings else -999)
	test_merged['mean_user_rating'] = test_merged.User.map(lambda x: mean_user_ratings[x] if x in mean_user_ratings else -999)

	train_merged['min_user_rating'] = train_merged.User.map(lambda x: min_user_ratings[x] if x in min_user_ratings else -999)
	test_merged['min_user_rating'] = test_merged.User.map(lambda x: min_user_ratings[x] if x in min_user_ratings else -999)

	train_merged['max_user_rating'] = train_merged.User.map(lambda x: max_user_ratings[x] if x in max_user_ratings else -999)
	test_merged['max_user_rating'] = test_merged.User.map(lambda x: max_user_ratings[x] if x in max_user_ratings else -999)

	train_merged['median_user_rating'] = train_merged.User.map(lambda x: median_user_ratings[x] if x in median_user_ratings else -999)
	test_merged['median_user_rating'] = test_merged.User.map(lambda x: median_user_ratings[x] if x in median_user_ratings else -999)

	train_merged['mean_artist_rating'] = train_merged.Artist.map(lambda x: mean_artist_ratings[x] if x in mean_artist_ratings else -999)
	test_merged['mean_artist_rating'] = test_merged.Artist.map(lambda x: mean_artist_ratings[x] if x in mean_artist_ratings else -999)

	train_merged['min_artist_rating'] = train_merged.Artist.map(lambda x: min_artist_ratings[x] if x in min_artist_ratings else -999)
	test_merged['min_artist_rating'] = test_merged.Artist.map(lambda x: min_artist_ratings[x] if x in min_artist_ratings else -999)

	train_merged['max_artist_rating'] = train_merged.Artist.map(lambda x: max_artist_ratings[x] if x in max_artist_ratings else -999)
	test_merged['max_artist_rating'] = test_merged.Artist.map(lambda x: max_artist_ratings[x] if x in max_artist_ratings else -999)

	train_merged['median_artist_rating'] = train_merged.Artist.map(lambda x: median_artist_ratings[x] if x in median_artist_ratings else -999)
	test_merged['median_artist_rating'] = test_merged.Artist.map(lambda x: median_artist_ratings[x] if x in median_artist_ratings else -999)

	train_merged = train_merged.fillna(-999)
	test_merged = test_merged.fillna(-999)

	train_merged['LIST_OWN'] = train_merged.LIST_OWN.astype(int)
	test_merged['LIST_OWN'] = test_merged.LIST_OWN.astype(int)

	train_merged['LIST_BACK'] = train_merged.LIST_BACK.astype(int)
	test_merged['LIST_BACK'] = test_merged.LIST_BACK.astype(int)

	return (train_merged, test_merged)

def feature_list(train_merged):
	return train_merged.columns.drop(['RESPID', 'Artist', 'User', 'Rating'])

def prepare_dataset(train_merged, test_merged, features):
	X = train_merged[features]
	y = train_merged.Rating

	final_test = test_merged[features]
	return X, y, final_test


def main(train, test, words, users):
	print('Feature Generation')
	train_merged, test_merged = merge_dataframes(train, test, words, users)
	train_merged, test_merged = generate_features(train, train_merged, test_merged)

	features = feature_list(train_merged)

	X, y, final_test = prepare_dataset(train_merged, test_merged, features)

	return X, y, final_test