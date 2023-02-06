import numpy as np
import time
import matplotlib.pyplot as plt

print("Exercice 1.4.1")
result = ""
for i in range(10):
    if(i<5) :
        result += "X"
    else :
        result=result[1:]
    print(result)
          
print("Exercice 1.4.2")
result = 0
input_str = "n45as29@#8ss6"
for i in range(len(input_str)):
    if input_str[i].isdigit():
        result += int(input_str[i])
print(result)

print('Exercice 1.4.3')
def convert_to_bin(inp1):
     
    if inp1 >= 1:
        convert_to_bin(inp1 // 2)
    print(inp1 % 2, end = '')
    
inp1 = input('Choose a number to convert to binary ')
convert_to_bin(int(inp1))

print('\nExercice 1.5-1')



def fibonaci(upper_threshold: int) -> list:
      if upper_threshold <= 0:
        return [0]
      sequence = [0, 1]
      while sequence[len(sequence)-1] < upper_threshold :
        next_value = sequence[len(sequence) - 1] + sequence[len(sequence) - 2]
        if next_value > upper_threshold :
            break
        sequence.append(next_value)
      return sequence

inp2 = input('Choose a number to enter in the fibonaci list ')
print(fibonaci(int(inp2)))

print("Exercice 1.5.2")

def display_as_digi(number: float) -> None:
    numbers = {
        '0':['xxx ', 'x x ', 'x x ', 'x x ', 'xxx '],
        '1':['  x ','  x ','  x ','  x '],
        '2':['xxx ','  x ', 'xxx ','x   ','xxx '],
        '3':['xxx ', '  x ', 'xxx ', '  x ','xxx '],
        '4':['x x ', 'x x ', 'xxx ', '  x ', '  x '],
        '5':['xxx ', 'x   ', 'xxx ', '  x ', 'xxx '],
        '6':['xxx ', 'x   ', 'xxx ', 'x x ', 'xxx '],
        '7':['xxx ', '  x ', '  x ', '  x ', '  x '],
        '8':['xxx ', 'x x ', 'xxx ', 'x x ', 'xxx '],
        '9':['xxx ', 'x x ', 'xxx ', '  x ', '  x '],
        ".":['   ', '   ', '   ', '   ', ' x ']
    }

    for line in range(5):
        line_str = ""
        for digit in str(number):
            line_str+=numbers[digit][line]
        print(line_str)

inp = input('Enter the number you want to display ')
display_as_digi(float(inp))


print("Exercice 2.1")


matrix = np.random.randint(0,25,size = (5,5))
print(matrix)

def matrix_threshold(threshold: int) -> None:
    for index, x in np.ndenumerate(matrix):
        if x < threshold :
            matrix[index]=0
        
    print(matrix)

def matrix_threshold_no_loop(threshold: int) -> None:
    matrix[matrix<threshold]=0
    print(matrix)

inp4 = input('Enter your threshold for the matrix  ')
t1=time.time()
matrix_threshold(int(inp4))
print("Time for function with loop : "+str(time.time()-t1))
t2=time.time()
matrix_threshold_no_loop(int(inp4))
print("Time for function without loop : "+str(time.time()-t2))


print("Exercice 2.2")

def show_in_digi(input_integer: float) -> None:
    numbersBool = {
                '1': [[False,False,True, False],[False,False,True, False],[False,False,True, False],[False,False,True, False],[False,False,True, False]],
                '2': [[True,True,True, False],[False,False,True, False],[True,True,True, False],[True,False,False, False],[True,True,True, False]],
                '3': [[True,True,True, False],[False,False,True, False],[True,True,True, False],[False,False,True, False],[True,True,True, False]],
                '4': [[True,False,True, False],[True,False,True, False],[True,True,True, False],[False,False,True, False],[False,False,True, False]],
                '5': [[True,True,True, False],[True,False,False, False],[True,True,True, False],[False,False,True, False],[True,True,True, False]],
                '6': [[True,True,True, False],[True,False,False, False],[True,True,True, False],[True,False,True, False],[True,True,True, False]],
                '7': [[True,True,True, False],[False,False,True, False],[False,False,True, False],[False,False,True, False],[False,False,True, False]],
                '8': [[True,True,True, False],[True,False,True, False],[True,True,True, False],[True,False,True, False],[True,True,True, False]],
                '9': [[True,True,True, False],[True,False,True, False],[True,True,True, False],[False,False,True, False],[False,False,True, False]],
                '0': [[True,True,True, False],[True,False,True, False],[True,False,True, False],[True,False,True, False],[True,True,True, False]],
                '.': [[False, False],[False, False],[False, False],[False, False],[True, False]]} 
    blank =  [[False,False],[False,False],[False,False],[False,False],[False,False]]
    display = blank
    for i in str(input_integer):
        display = np.concatenate((display,numbersBool[i]), axis = 1)
        display = np.concatenate((display,blank), axis = 1)
        

    plt.imshow(display,cmap= 'binary')
    plt.show()
    pass
inp5 = input('Enter the number you want to display ')
show_in_digi(float(inp5))

