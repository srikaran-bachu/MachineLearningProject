import copy
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()
model = load_model('Models/NN_Models/Trained-Model-ML-1699315388.285516')
#model = load_model('/Users/kunalnakka/CS Files/ML Class/Comparisons/NBA-Machine-Learning-Sports-Betting/Models/Trained-Model-ML-1701289840.380623')

def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(model.predict(np.array([row])))
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            print('The winner for ' + home_team + " vs " + away_team + ' is ' + home_team + " the win chance is"+  f" ({winner_confidence}%)")
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            print('The winner for ' + home_team + " vs " + away_team + ' is ' + away_team + " the win chance is"+  f" ({winner_confidence}%)")
        count += 1
    # print(ml_predictions_array)

    deinit()
