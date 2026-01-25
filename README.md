upcoming addtions


1. text to lower case 
2. remove punctuation
3. tokenization
4. remove stop words 
5. stemming 

tokenization -> break text into individual words 
stop words -> doesnt add meaning to the text ~ like the, a , this , that 
stemming -> running ~ run , amazingly ~ amazing 


Usuage of translate(), maketrans():

translate can be used to replace or delete characters with the use of a translation table.
maketrans is used to make a translation table.

new_string = old_string.translate(table)
table = str.maketrans("123", "abc") # repalces 1 with a , 2 with b and so on.
table = str.maketrans("", "", "123") # deletes the instances of 1, 2 and 3 

str.maketrans(x, y, z)
x → characters to replace
Sy → what to replace them with
z → characters to DELETE

string.punctuation returns all the punctuation characters 
!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~





Stemming :
running, runs, ran, run ---> run 
==> using nltk.stem library ==> PorterStemmer class 
