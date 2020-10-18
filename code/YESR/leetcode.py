# water = [5,2,1,2,1,5]
# A,B = 0,0
# i = -1
# max_water = 0
# while B<len(water)-1:
#     i = i + 1
#     if water[i] > water[i+1]:
#         A,B = i,i+1
#         # if B+1<len(water):
#         while (water[B+1] - water[B]<=0):
#             B = B+1
#             if B+1 >= len(water):
#                 break
#         # print(B)
#         # if B+1 < len(water):
#         while (water[B+1] -water[B]>=0):
#             B = B+1
#             if B+1 >= len(water):
#                 break
#         print("l:"+str(A),"r:"+str(B))
#         hight = min(water[A],water[B])
#         for j in range(A,B):
#             if water[j]<hight:
#                 max_water = hight-water[j]+max_water
#         A = B
#         i = A-1
# print(max_water)\

def parenthesis(string):
    temp = list()
    symbol = ["(",")", "{","}", "[","]"]
    for i,j in enumerate(string):
        if len(temp) == 0:
            temp.append(j)
        else:
            idx_1 = symbol.index(j)
            idx_2 = symbol.index(temp[-1])
            if idx_2+1 == idx_1:
                temp.pop(-1)
    # if len(temp)== 0

# run out of time
def coinChange(coins, amount):
    if amount == 0:
        return 0
    for i in range(len(coins)):
        if coins[i] == amount:
            return 1
        if amount < 0:
            return -1
    max_count = coinChange(coins, amount - coins[0])
    for i in range(1, len(coins)):
        if 0 < coinChange(coins, amount - coins[i]) < max_count:
            max_count = coinChange(coins, amount - coins[i])
    if max_count + 1 <= 0:
        return -1
    else:
        return max_count + 1
#print(coinChange([1,2,11],5))

# dp array
def change_coin2(coin,amount):
    dp = [float("inf")]*(amount+1)
    dp[0] = 0
    for c in coin:
        for i in range(c,amount+1):
            dp[i] = min(dp[i],dp[i-coin]+1)
    return dp[amount] if dp[amount] != float("inf") else -1
# print(change_coin2([1,2,11],5))

# listA = [1,[2,3,4],5]
# listB = copy.deepcopu
# listB[1][0]=8
# print(listA)

def stack_edit():
    print("please input:")
    s = input()
    result = []
    for i in s:
        if len(result)>0:
            if result[-1] == "(":
                if i == "(":
                    result.append(i)
                if i == ")":
                    result.pop()
                continue
        if i == "<":
            result.pop()
            continue
        result.append(i)
    print(''.join(result))


def LIS(nums):
    print("please input:")
    # s = input()
    s = nums
    n = len(s)
    dp = n*[1]
    for i in range(n):
        for j in range(i):
            if s[i] > s[j]:
                dp[i] = max(dp[j] + 1,dp[i])
                print(dp)
    return max(dp) if n != 0 else 0

# out = LIS([1,3,6,7,9,4,10,5,6])
# print(out)

def test_unpack(a,b,c):
    print(a,b,c)

# a = [2,3,4]
# test_unpack(*a)

class Solution:
    def key_and_room(self, rooms):
        n = len(rooms)
        flag = [0] * n
        flag[0] = 1
        self.enter_room(rooms, rooms[0], flag)
        print(flag)
        return True if sum(flag) == n else False


    def enter_room(self, rooms, keys, flag):
        for key in keys:
            if flag[key] == 1:
                continue
            flag[key] = 1
            self.enter_room(rooms, rooms[key], flag)

# so = Solution()
# print(so.key_and_room([[1,3],[1,4],[2,3,2,4,1],[],[4,3,2]]))MMMM
# MMFF
# FMMM
# MMMM
# MFMM


def violate_solution():
    m,n = list(map(int,input().split()))
    array = []
    for i in range(m):
        array.append(list(input()))


    def find_max(array, i, j, m, n):
        max_n = 0
        max_m = 0
        while(i<n):
            if array[i] == "F":
                break
            max_n += 1
            i += 1
        while(j<m):
            if array[j] == "F":
                break
            max_m += 1
            j = j+1
        return min(max_n,max_m)


    visit = [[0]*n for _ in range(m)]
    max_s = 0
    for i in range(m):
        for j in range(n):
            if visit[i] == 1:
                continue
            visit[i][j] = 1
            max_s = max(max_s,find_max(array,i, j, m,n))
    print(max_s)


def max_squre(array):
    '''
    dp[i][j] = max(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])+1
    :param array:
    :return:
    '''
    m,n = len(array), len(array[0])
    dp = [[0]*n for _ in range(m)]
    # bounding problem
    for i in range(m):
        if array[i][0] == "M":
            dp[i][0] = 1

    for j in range(n):
        if array[0][j] == "M":
            dp[0][j] = 1

    for i in range(m):
        for j in range(n):
            if (array[i][j] == "F"):
                continue
            dp[i][j] = max(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return max(dp)**2

#print(max_squre())
a = [[4,2],[2,30]]
print(max(a))