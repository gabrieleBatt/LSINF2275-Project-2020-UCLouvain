import sys


def dump(type, obj):
    print("-------------------------------------------------------------------------------------------")
    print("*[" + type + "]:\n", obj)


# used to check the probability matrices
def consistency_check(s_dice, r_dice, id):
    check_successful = True;
    for j in range(0, len(s_dice.T[0])):
        if (1 - (sum(s_dice[j])) > 1e-7):
            dump("Safety row %d" % j, s_dice[j])
            check_successful = False
    for j in range(0, len(r_dice.T[0])):
        if (1 - (sum(r_dice[j])) > 1e-7):
            dump("Risky row %d" % j, r_dice[j])
            check_successful = False
    if (check_successful):
        dump("Consistency check successful", id)
    else:
        dump("Consistency check failed", id)
