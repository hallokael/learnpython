from functools import reduce
def normalize( s ):
    return s[:1].upper()+s[1:].lower()
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)
def prod(L):
    return reduce( lambda x,y:x*y,L )
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
def str2float(s):
    def ch2num(s):
        if s!='.':
            return int( s )
        else: return s
    def Mul( x,y ):
        return x*10+y if y!='.' else x
    return reduce( Mul,map( ch2num,s ) )/10**( len( s )-s.find('.')-1 )
print('str2float(\'12345.678\') =', str2float('12345.678'))



A=[36, 5, -12, 9, -21]
print(sorted( A,key=abs ))
print(sorted( A,key=lambda x:-x if x<0 else x))
print( sorted( ['bob','about','Zoo','Credit'],key=str.lower ) )


L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
def by_score( t ):
    return -t[1]
L2 = sorted(L, key=by_name)
print(L2)

L2 = sorted(L, key=by_score)
print(L2)
