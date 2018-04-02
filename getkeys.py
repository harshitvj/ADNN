# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def keys_to_output(keys):
    #        [A, W, D, S, AW, DW, AS, DS]
    output = [0, 0, 0, 0, 0, 0, 0, 0]

    if 'A' in keys:
        if 'W' in keys:
            output[4] = 1
        elif 'D' in keys:
            output[6] = 1
        else:
            output[0] = 1

    if 'W' in keys:
        if 'A' in keys:
            output[4] = 1
        elif 'D' in keys:
            output[5] = 1
        else:
            output[1] = 1

    if 'D' in keys:
        if 'W' in keys:
            output[5] = 1
        elif 'D' in keys:
            output[7] = 1
        else:
            output[2] = 1

    if 'S' in keys:
        if 'A' in keys:
            output[6] = 1
        elif 'D' in keys:
            output[7] = 1
        else:
            output[3] = 1

    return output
