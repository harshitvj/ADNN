from directkeys import PressKey, ReleaseKey, W, A, S, D


#        [A, W, D, AW, DW]
#        [0, 1, 2, 3, 4]

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
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
        elif op == 5:
            PressKey(W)
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
        else:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
