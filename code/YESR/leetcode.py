water = [5,2,1,2,1,5]
A,B = 0,0
i = -1
max_water = 0
while B<len(water)-1:
    i = i + 1
    if water[i] > water[i+1]:
        A,B = i,i+1
        # if B+1<len(water):
        while (water[B+1] - water[B]<=0):
            B = B+1
            if B+1 >= len(water):
                break
        # print(B)
        # if B+1 < len(water):
        while (water[B+1] -water[B]>=0):
            B = B+1
            if B+1 >= len(water):
                break
        print("l:"+str(A),"r:"+str(B))
        hight = min(water[A],water[B])
        for j in range(A,B):
            if water[j]<hight:
                max_water = hight-water[j]+max_water
        A = B
        i = A-1
print(max_water)