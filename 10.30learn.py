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
            print(s,"zzz")
            return int( s )
        else: return s
       # return int( s) if s!='.' else s
    def Mul( x,y ):
        return x*10+y if y!='.' else x
    return reduce( Mul,map( ch2num,s ) )/10**( len( s )-s.find('.')-1 )

print('str2float(\'12345.678\') =', str2float('12345.678'))