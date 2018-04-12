from directkeys import PressKey, ReleaseKey, W, A, S, D


#        [A, W, D, S, AW, DW, AS, DS, NothingPressed]

class Controller:
    @staticmethod
    def act(op):
        if op == 0:
            PressKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
        elif op == 1:
            PressKey(W)
            ReleaseKey(A)
            ReleaseKey(D)
            ReleaseKey(S)
        elif op == 2:
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(D)
            ReleaseKey(W)
        elif op == 3:
            PressKey(S)
            ReleaseKey(A)
            ReleaseKey(D)
            ReleaseKey(W)
        elif op == 4:
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
        elif op == 5:
            PressKey(W)
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
        elif op == 6:
            PressKey(A)
            PressKey(S)
            ReleaseKey(D)
            ReleaseKey(W)
        elif op == 7:
            PressKey(D)
            PressKey(S)
            ReleaseKey(A)
            ReleaseKey(W)
        else:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
