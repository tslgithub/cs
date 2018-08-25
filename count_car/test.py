import operator
a = [[1,2],[3,4],[5,6],[1,2],[1,2],[3,4],[5,6],[1,2] ]
rs = map(set,a)
print( list(rs) )  
print('----------------------------')
k,j=0,0
for item in a:
    k+=1
    for item2 in a[k:]:
        j+=1
        if operator.eq(item,item2):
            a.remove(a[j])
        
        if k == len(a)-1:
            break
print (a)
pritn('Done')
