import re
print(re.match(r'^\d{3}\-\d{3,8}$','010-12345'))
print( 'a b    sd fer'.split(' ') )
print( re.split(r'\s+','a b   sd fer') )
print( re.split(r'[\s,;:]+','a,b:c::,;d  e;dsa  a') )
m=re.match(r'^(\d{3})-(\d{3,8})$','010-12345')
print(m.group(0),m.group(1),m.group(2))
re_a=re.compile(r'^[1-9]+$')
print(re_a.match('3242'))