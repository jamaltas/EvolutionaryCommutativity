from landscape_evolution_plot_combin import *         #Import the functions
import seaborn as sns
import pprint
from numpy.linalg import matrix_rank
from numpy.linalg import norm
import statistics
import random
from scipy.stats.stats import pearsonr


def main():

    N = 6
    sigma = 1
    landscapes = 50

    commutativity = [[] for i in range(40)]
    magCommutativity = [[] for i in range(40)]
    rowsZero = [[] for i in range(40)]

    """
    Run the simulation many times for mean behavior.
    """
    count = 0
    while count < 200:
        for kk in range(landscapes):

            """
            Generate two landscapes A and B in the same statistical fashion
            """
            A = Landscape(N, sigma)
            B = Landscape(N, sigma)

            """
            Make the B landscape 100% anticorrelated with A. Then shuffle it "kk" times to make things less and less perfectly anticorrelated.
            """
            antiB = A.get_anticorrelated_landscape(B)
            antiB = antiB.generate_shuffled_landscape(kk)


            """
            Make the B landscape 100% correlated with A. Then shuffled it "kk" times to make it less and less perfectly correlated.
            """
            corrB = A.get_correlated_landscape(B)
            corrB = corrB.generate_shuffled_landscape(kk)

            """
            Bin all the results in 40 equally spaced bins, calculate what bin the above landscapes fall into based off their correlation
            """
            bin1 = int((pearsonr(A.ls,corrB.ls)[0] + 1)/.05)   # What list should i store the values in for corrBs
            bin2 = int((pearsonr(A.ls,antiB.ls)[0] + 1)/.05)   # What list should i store the values in for AntiBs

            if bin1 == 40:
                bin1 -=1

            if bin2 == 40:
                bin2 -= 1


            """
            Get the Transition Matrices for A and both B matrices we've created.
            """
            ATM = A.get_TM()
            BantiTM = antiB.get_TM()
            BcorrTM = corrB.get_TM()


            """
            Below calculates # of rows of the commutator with all 0s. Ignore the transpose, it's just to make them right stochastic matrices which I find easier to think about.
            """
            Xanti = np.dot(ATM.transpose(),BantiTM.transpose())
            Xcorr = np.dot(ATM.transpose(),BcorrTM.transpose())

            Yanti = np.dot(BantiTM.transpose(), ATM.transpose())
            Ycorr = np.dot(BcorrTM.transpose(), ATM.transpose())

            antiRowZ = 0
            corrRowZ = 0
            for z in range(2**N):
                if np.all((Xanti-Yanti)[z][:] == 0):
                    antiRowZ += 1
                if np.all((Xcorr-Ycorr)[z][:] == 0):
                    corrRowZ += 1

            rowsZero[bin1].append(corrRowZ/(2**N))
            rowsZero[bin2].append(antiRowZ/(2**N))


            """
            Below checks each genotype for commutativity by evoling in each landscape.
            """
            commAnti = 0
            commCorr = 0
            for z in range(2**N):
                p0 = np.zeros((2**N,1))
                p0[z][0] = 1
                ABanti = np.dot(np.linalg.matrix_power(ATM, 1), p0)
                ABanti = np.dot(np.linalg.matrix_power(BantiTM, 1), ABanti)

                BAanti = np.dot(np.linalg.matrix_power(BantiTM, 1), p0)
                BAanti = np.dot(np.linalg.matrix_power(ATM, 1), BAanti)


                ABcorr = np.dot(np.linalg.matrix_power(ATM, 1), p0)
                ABcorr = np.dot(np.linalg.matrix_power(BcorrTM, 1), ABcorr)

                BAcorr = np.dot(np.linalg.matrix_power(BcorrTM, 1), p0)
                BAcorr = np.dot(np.linalg.matrix_power(ATM, 1), BAcorr)

                if np.all(ABanti == BAanti):
                    commAnti += 1

                if np.all(ABcorr == BAcorr):
                    commCorr += 1

            commutativity[bin1].append(commCorr/(2**N))
            commutativity[bin2].append(commAnti/(2**N))


            """
            Below calculates the norm of the commutator for "imperfect commutativity"
            """
            magCommutativity[bin1].append(norm(ABcorr-BAcorr))
            magCommutativity[bin2].append(norm(ABanti-BAanti))


        count += 1

        if count % 10 == 0:
            print(count)




    """
    Below plots the above data.
    """
    x = np.linspace(-.975,.975,40)
    avgCommutativity = []
    stdCommutativity = []
    numCommutativity = []

    avgMagComm = []
    avgRowZero = []

    for i in range(40):
        avgCommutativity.append(statistics.mean(commutativity[i]))
        avgMagComm.append(statistics.mean(magCommutativity[i]))
        avgRowZero.append(statistics.mean(rowsZero[i]))

        stdCommutativity.append(statistics.pstdev(commutativity[i]))
        numCommutativity.append(len(commutativity[i]))

    print(numCommutativity)

    plt.plot(x, avgCommutativity)
    plt.plot(x, avgMagComm)
    plt.plot(x, avgRowZero)

    plt.figure()
    plt.scatter(avgCommutativity, avgRowZero)
    plt.show()


if __name__ == '__main__':
    main()
