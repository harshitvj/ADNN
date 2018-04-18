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
        elif op == 1:
            PressKey(W)
            ReleaseKey(A)
            ReleaseKey(D)
        elif op == 2:
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(W)
        elif op == 3:
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)
        elif op == 4:
            PressKey(D)
            PressKey(W)
            ReleaseKey(A)
        else:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(D)
