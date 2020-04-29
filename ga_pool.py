import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()


class pool(object):
    """docstring for pool"""

    def __init__(self, dim, poolsize, actionlist_uinp, initval, targetval):
        super(pool, self).__init__()
        self.dim = dim
        self.poolsize = poolsize
        self.poolarray = np.array([])
        self.fitnessmap = np.array([])
        self.actions = actions(actionlist_uinp)
        self.actionlist = self.actions.actionlist

        self.initval = initval
        self.target = targetval

    def createpool(self):
        poolarray = []
        for i in range(self.poolsize):
            poolarray.append(gene(self.dim, len(self.actionlist)))
        self.poolarray = poolarray

    def mutate(self, geneit):
        randind = np.random.choice(self.dim)
        geneit.geneit[randind] = np.random.choice(range(0, len(self.actionlist)))
        return geneit

    def crossover(self, genea, geneb):
        randind = np.random.choice(self.dim)
        parta = genea.geneit[:randind]
        partb = geneb.geneit[:randind]

        genea.geneit[:randind] = partb
        geneb.geneit[:randind] = parta
        return genea, geneb

    def createnew(self):
        return gene(self.dim).geneit

    def fitness(self):

        # result = self.actions.decode(self.initval,gene.geneit)
        count = 0
        length = len(self.poolarray)
        fitnessmap = []
        for gene in self.poolarray:
            count += 1
            result = self.actions.decode(self.initval, gene.geneit)

            error = abs(int(self.target) - int(result))
            # print(result,self.target,error)
            print(100 * (count / length))
            fitnessmap.append(error)

        self.fitnessmap = fitnessmap

    def calc_fitness(self):
        errors = Parallel(n_jobs=num_cores)(delayed(self.fitness)(i) for i in tqdm(self.poolarray))
        self.fitnessmap = errors

    def passnextgen(self):

        self.fitness()

        numelit = 2

        prob_motation = 0.6
        prob_crossover = 0.4
        prob_new = 0.2

        nextpool = []
        elit_indx = np.argpartition(self.fitnessmap, numelit)[:numelit]

        for i in elit_indx:
            nextpool.append(deepcopy(self.poolarray[i]))
        # print(elit_indx)

        prob = np.random.rand()

        for ind, cgene in enumerate(self.poolarray):
            # cgene = deepcopy(cgene)

            if ind in elit_indx:
                # print('elit')

                continue

            else:
                # print('else')

                if prob < prob_motation:
                    # print('mut')
                    cgene = self.mutate(cgene)
                if prob < prob_crossover:
                    othergene = np.random.choice(self.poolarray)
                    cgene, othergene = self.crossover(cgene, othergene)
                if prob < prob_new:
                    cgene = gene(self.dim, len(self.actionlist))

            nextpool.append(cgene)

        self.poolarray = nextpool[0:self.poolsize]


class gene(object):
    """docstring for gene"""

    def __init__(self, dim, actionsize):
        super(gene, self).__init__()
        self.dim = dim
        self.actionsize = actionsize
        self.geneit = self.create_gene()

    def create_gene(self):
        # return np.array([4,0,3,1,4,3,1,0,2])
        return np.random.choice(range(0, self.actionsize), self.dim)


class actions(object):
    """docstring for actions"""

    def __init__(self, actionlist):
        super(actions, self).__init__()

        for ind, i in enumerate(actionlist):
            if i[0] == 'stuck':
                actionlist.append(['stuckpress', ind])
        self.actionlist = actionlist
        self.dynamicactionlist = deepcopy(self.actionlist)

    def SUM(self, cval):
        if cval[0] == '-':
            negative = True
            cval = cval[1:]
        else:
            negative = False

        res = 0
        for i in cval:
            res += int(i)
        if negative:
            res = -res
        return (str(res))

    def add(self, cval, val):
        return str(int(cval) + int(val))

    def reverse(self, cval):
        if cval[0] == '-':
            negative = True
            cval = cval[1:]
        else:
            negative = False
        midval = cval[::-1]

        if negative:
            midval = '-' + midval
        return midval

    def multiply(self, cval, val):
        return (str(round(int(cval) * float(val))))

    def appendn(self, cval, val):
        length = len(val)
        multiplier = '1e+' + str(length)
        res = float(multiplier) * int(cval) + int(val)
        # return str(round(res))
        if len(val) == 1:
            return str(10 * int(cval) + int(val))
        elif len(val) == 2:
            return str(100 * int(cval) + int(val))
        elif len(val) == 3:
            return str(1000 * int(cval) + int(val))

    def construct(self, actionlist):
        return

    def stuck(self, cval, val):
        if val == '':
            return cval
        else:
            res = self.appendn(cval, val)
            return res

    def stuckpress(self, cval, ind):
        self.dynamicactionlist[ind][1] = cval

    def changesign(self, cval):
        if cval[0] == '-':
            return cval[1:]
        else:
            return '-' + cval

    def power(self, cval, powerval):
        return str(int(cval) ** int(powerval))

    def inc_adds(self, mag):
        for ind, i in enumerate(self.dynamicactionlist):
            if i[0] in ('add', 'appendn', 'multiply', 'divide'):
                ispositive = 0 < int(self.dynamicactionlist[ind][1])
                if ispositive:
                    self.dynamicactionlist[ind][1] = str(int(self.dynamicactionlist[ind][1]) + int(mag))
                else:
                    self.dynamicactionlist[ind][1] = str(int(self.dynamicactionlist[ind][1]) - int(mag))

    def transform(self, cval, bef, aft):
        cval = cval.replace(bef, aft)
        return cval

    def erase(self, cval):
        if len(cval) == 1:
            return '0'
        elif len(cval) == 2 and cval[0] == '-':
            return '0'
        else:
            return cval[:-1]

    def inverse10(self, cval):
        new_num = ''
        for ind, i in enumerate(cval):
            if i in [str(num) for num in range(1, 10)]:
                print('i in', cval, i)
                replaced = str(10 - int(i))
                new_num += replaced
                print(new_num)
            else:
                new_num += i
        return new_num

    def portal(self, cval):  # 29910 29 910
        if cval[0] == '-':
            return cval

        if len(cval) >= portal[1]:
            rev_portal = [0, 0]
            rev_portal[0] = len(cval) - portal[0]
            rev_portal[1] = len(cval) - portal[1]

            multiplier = portal[0] - 1
            number = '0'
            if rev_portal[1] >= 1:
                frontnums = [float('1e+' + str(multiplier)) * int(cval[i]) for i in range(0, rev_portal[1] + 1)]
                for frontnum in frontnums:
                    number = str(int(number) + int(frontnum))
                number = str(int(number) + int(cval[rev_portal[1] + 1:]))
                print('frontnums', frontnums)
            else:
                frontnum = (float('1e+' + str(multiplier)) * int(cval[rev_portal[1]]))
                number = str(int(frontnum) + int(cval[rev_portal[1] + 1:]))
            # print('rew',rev_portal,portal,'fn',frontnum,cval[rev_portal[1]+1:],cval)

            print(number)
            if len(number) >= portal[1]:
                number = self.portal(number)
            return number
        else:
            return cval

    def shift(self, cval, direction):
        if cval[0] == '-':
            negative = True
            cval = cval[1:]
        else:
            negative = False

        if direction == 'l':
            midval = cval[1:]
            midval = midval + cval[0]
        elif direction == 'r':
            midval = cval[:-1]
            midval = cval[-1] + midval
        midval = int(midval)
        if negative:
            midval = -midval
        return (str(midval))

    def mirror(self, cval):
        if cval[0] == '-':
            negative = True
            cval = cval[1:]
        else:
            negative = False

        midval = cval + cval[::-1]
        midval = int(midval)
        if negative:
            midval = -midval
        return (str(midval))

    def decode(self, initval, numbers):
        cval = initval
        self.dynamicactionlist = deepcopy(self.actionlist)
        for i in numbers:
            # print(cval,i)
            currentop = self.dynamicactionlist[i][0]
            if currentop == 'sum':
                cval = self.SUM(cval)
            elif currentop == 'reverse':
                cval = self.reverse(cval)
            elif currentop == 'add':
                cval = self.add(cval, self.dynamicactionlist[i][1])
            elif currentop == 'multiply':
                cval = self.multiply(cval, self.dynamicactionlist[i][1])
            elif currentop == 'divide':
                if int(cval) % int(self.dynamicactionlist[i][1]) != 0:
                    return '9999999'
                else:
                    cval = self.multiply(cval, 1 / int(self.dynamicactionlist[i][1]))
            elif currentop == 'transform':
                cval = self.transform(cval, self.dynamicactionlist[i][1], self.dynamicactionlist[i][2])
            elif currentop == 'appendn':
                cval = self.appendn(cval, self.dynamicactionlist[i][1])
            elif currentop == 'changesign':
                cval = self.changesign(cval)
            elif currentop == 'power':
                cval = self.power(cval, self.dynamicactionlist[i][1])
            elif currentop == 'shift':
                cval = self.shift(cval, self.dynamicactionlist[i][1])
            elif currentop == 'mirror':
                cval = self.mirror(cval)
            elif currentop == 'erase':
                cval = self.erase(cval)
            elif currentop == 'stuck':
                cval = self.stuck(cval, self.dynamicactionlist[i][1])
            elif currentop == 'stuckpress':
                self.stuckpress(cval, self.dynamicactionlist[i][1])
            elif currentop == 'inc_adds':
                self.inc_adds(self.dynamicactionlist[i][1])
            elif currentop == 'inverse10':
                cval = self.inverse10(cval)
            else:
                print('error', currentop)

            if len(cval) > 7:  #

                return '9999999'
            cval = self.portal(cval)
        return cval


"""
a = pool(5,50)

a.createpool()
for i in range(2000):
	a.passnextgen()
	print(min(a.fitnessmap))
print(a.poolarray[0].geneit)
"""

poolsize = 100
iters = 10

moves = 4
portal = [1, 5]

init_val = '22'
target_val = '116'
# actions_inp = [['appendn','1'],['shift','r'], ['appendn','9'],['transform','89','99']] #['multiply','4'],
# actions_inp = [['multiply','3'],['appendn','9'],['shift','r']]

# 101
# actions_inp = [['appendn','1'],['transform','15','51'],['multiply','5'],['sum']]

# 123
actions_inp = [['appendn', '6'], ['add', '-3'], ['mirror'], ['sum']]

a = pool(moves, poolsize, actions_inp, init_val, target_val)
a.createpool()

for i in range(iters):
    # print(a.poolarray[0].geneit)
    a.passnextgen()
    print('passed', "%.2f" % float((i / iters) * 100), 'percent')
    print('min. error', min(a.fitnessmap))
    if (min(a.fitnessmap)) < 0.00001:
        a.fitness()
        # pass
        print('already found the soln!')
        break

bestindx = np.squeeze(np.argpartition(a.fitnessmap, 1)[:1])

bestgene = a.poolarray[bestindx].geneit

print('\nsolution is:\n\n')

for i in bestgene:
    print(actions_inp[i])

a = actions(actions_inp)
print('act', a.decode('299', [0]))
# 29910 2919 921
