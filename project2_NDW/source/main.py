from time import time

import numpy as np

data = np.zeros([100000, 4], dtype=int)
data2 = np.zeros([100000, 22], dtype=int)
genre = np.zeros([19, 2], dtype='<U256')
item = np.zeros([1682, 24], dtype='<U256')
user = np.zeros([943, 5], dtype='<U256')
user2 = np.zeros([943, 24], dtype=int)
ratings = np.zeros([943, 1682], dtype=int)


def main():
    # opening the files
    fdata = open("..\\resources\\ml-100k\\u.data", "r")
    fgenre = open("..\\resources\\ml-100k\\u.genre", "r")
    fitem = open("..\\resources\\ml-100k\\u.item", "r")
    fu = open("..\\resources\\ml-100k\\u.user", "r")

    # extract the data
    extractDataInt(data, fdata, '	')
    extractDataString(genre, fgenre, '|')
    extractDataString(item, fitem, '|')
    extractDataString(user, fu, '|')

    # closing the files
    fdata.close()
    fgenre.close()
    fitem.close()
    fu.close()

    for i in range(1, 944):
        tmp = np.asarray(userReviews(i), dtype=int)
        for x in tmp:
            ratings[i - 1][x[1] - 1] = x[2]

    # makes an array containing the different occupations
    occupations = np.unique(user[:, 3])
    occupations.astype(dtype='<U256')

    # create a summary array of users with binary representation of occupations
    user2[:, :2] = user[:, :2]

    # gender is also in binary representation
    for i in range(0, 943):
        if user[i, 2] == 'M':
            user2[i, 2] = 0
        else:
            user2[i, 2] = 1

    for i in range(0, len(user2)):
        index = np.where(user[i, 3] == occupations)
        user2[i, 3 + index[0][0]] = 1

    # create a summary array of data with the genres of the film reviewed
    data2[:, :3] = data[:, :3]

    for i in range(0, len(data2)):
        data2[i, 3:] = item[data2[i, 1] - 1, 5:]

    """"pearsonVal = np.zeros((943, 1682, 2))
    for u in range(0, 943):
        for i in range(0, 1682):
            pearson0 = pearsonArray(u, i)
            pearson0 = pearson0[np.argsort(pearson0[:, 0])]
            pearsonFiltered = pearsonFilter(u, pearson0, i, 3)
"""
    a, c = np.unique(ratings, return_counts=True)
    print(a)
    print(c)

    for u in range(0, 943):
        print(u)
        zeros = np.where(ratings[u, :] == 0)
        for i in zeros[0]:
            print(i)
            pearson0 = pearsonArray(u, i)
            pearson0 = pearson0[np.argsort(pearson0[:, 0])]
            pearsonFiltered = pearsonFilter(u, pearson0, i, 3)
            tmp = predictRate(u, i, pearsonFiltered, len(pearsonFiltered))
            if np.isnan(tmp):
                ratings[u, i] = -1
            else:
                ratings[u, i] = tmp

    a, c = np.unique(ratings, return_counts=True)
    print(a)
    print(c)


def extractDataString(matrix, fd, separator):
    f1 = fd.readline()
    row = len(matrix)
    column = len(matrix[0])

    for i in range(0, row):
        f2 = f1.split(separator, column)
        f2[column - 1] = f2[column - 1].replace('\n', '')
        for j in range(0, column):
            matrix[i][j] = f2[j]
        f1 = fd.readline()


def extractDataInt(matrix, fd, separator):
    f1 = fd.readline()
    row = len(matrix)
    column = len(matrix[0])

    for i in range(0, row):
        f2 = f1.split(separator, column)
        f2[column - 1] = f2[column - 1].replace('\n', '')
        for j in range(0, column):
            f2[j] = int(f2[j], 10)
            matrix[i][j] = f2[j]
        f1 = fd.readline()


def getTitle(filmId):
    return item[filmId - 1, 1]


def filmReviews(filmId):
    return data[(data[:, 1] == filmId)]


def userReviews(userId):
    return data[(data[:, 0] == userId)]


def meanUser(u):
    listReviews = np.asarray(userReviews(u), dtype=int)[:, 2]
    return listReviews.sum() / len(listReviews)


def reviewsIntersect(u, v):
    a = userReviews(u+1)[:, 1]
    b = userReviews(v+1)[:, 1]
    c = np.intersect1d(a, b)
    d = np.zeros((len(c), 2))
    for i in range(0, len(c)):
        d[i, 0] = ratings[u, c[i]-1]
        d[i, 1] = ratings[v, c[i]-1]
    return d


def pearsonFilter(u, pearson, it, k):
    i = 0
    a = []
    for x in pearson[::-1]:
        if i == k:
            break
        d = int(x[1])-1
        if ratings[d, it] > 0 and d-1 != u:
            b = [x[0], x[1]]
            a.append(b)
            i = i+1
    return np.asarray(a, dtype=float)


def pearsonValue(u, v):
    inter = reviewsIntersect(u, v)

    if inter.size == 0:
        return 0
    mean = inter.mean(0)
    uu = mean[0]
    uv = mean[1]
    s = 0
    su2 = 0
    sv2 = 0

    for i in range(0, len(inter)):
        s += (inter[i][0] - uu) * (inter[i][1] - uv)
        su2 += (inter[i][0] - uu) * (inter[i][0] - uu)
        sv2 += (inter[i][1] - uv) * (inter[i][1] - uv)

    if su2 == 0 or sv2 == 0:
        return -1
    return s / (np.sqrt(su2) * np.sqrt(sv2))


def pearsonArray(u, it):
    tmp = filmReviews(it+1)
    pearson0 = np.zeros((len(tmp), 2), 'float64')
    for i in range(0, len(tmp)):
        pearson0[i, 1] = tmp[i, 0]
        pearson0[i, 0] = pearsonValue(u, tmp[i, 0]-1)

        if pearson0[i, 0] > 1.0:
            pearson0[i, 0] = 1.0
        elif pearson0[i, 0] < -1.0:
            pearson0[i, 0] = -1.0
    #print(pearson0)
    #print(np.where(np.isclose(pearson0[:, 0], 1, 1e-5)))
    return pearson0


def predictRate(u, i, pearson, k):
    a = pearson
    num = 0.0
    den = 0.0
    for j in range(0, k):
        id = int(a[j, 1])
        num += a[j, 0] * (ratings[id-1, i] - meanUser(id))
        den += np.abs(a[j, 0])
    return np.round(meanUser(u+1) + num/den)


if __name__ == '__main__':
    main()
