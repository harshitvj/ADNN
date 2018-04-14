import numpy as np
from collections import Counter
import pandas as pd
from random import shuffle

filename = 'training_data.npy'
path_balanced_data = 'balanced_data.npy'


def balance():
    try:
        training_data = list(np.load(filename))
    except FileNotFoundError:
        print('File not Found')
        training_data = []
        exit(404)

    df = pd.DataFrame(training_data)
    counter_obj = Counter(df[1].apply(str))
    length = min(counter_obj.values())

    print('Min length:', length)
    # length = 1000

    forward = []
    left = []
    right = []
    backward = []
    forward_left = []
    forward_right = []
    no_key = []

    #        [A, W, D, S, AW, DW, No_Key]

    for data in training_data:
        img = data[0]
        choice = data[1]
        if choice == [1, 0, 0, 0, 0, 0, 0]:
            left.append([img, choice])
        elif choice == [0, 1, 0, 0, 0, 0, 0]:
            forward.append([img, choice])
        elif choice == [0, 0, 1, 0, 0, 0, 0]:
            right.append([img, choice])
        elif choice == [0, 0, 0, 1, 0, 0, 0]:
            backward.append([img, choice])
        elif choice == [0, 0, 0, 0, 1, 0, 0]:
            forward_left.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 1, 0]:
            forward_right.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 0, 1]:
            no_key.append([img, choice])
        else:
            print('wut?')

    forward = forward[:length]
    backward = backward[:length]
    left = left[:length]
    right = right[:length]
    forward_left = forward_left[:length]
    forward_right = forward_right[:length]
    no_key = no_key[:length]

    balanced_data = forward + backward + left + right + forward_left + forward_right + no_key

    shuffle(balanced_data)
    print('Saving Balanced data in', path_balanced_data, 'Length:', len(balanced_data))
    np.save('balanced_data.npy', balanced_data)


balance()
