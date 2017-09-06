import sys


coef = sys.argv

order_1_file = coef[1]
order_2_file = coef[2]

def combine_order_1_2(order_1_file,order_2_file):

    f1 = open(order_1_file,'r')
    f2 = open(order_2_file,'r')

    combine_file_name = order_1_file.split('_1_')[0]+'_combine_dim_' + str(2*int(order_1_file.split('_1_dim_')[1]))

    f3 = open(combine_file_name,'w')

    f1.readline()
    f2.readline()

    for line1 , line2 in zip(f1.readlines(),f2.readlines()):
        line = line1.strip() + ' ' + ' '.join(line2.strip().split(' ')[1:]) + '\n'
        f3.write(line)

    f1.close()
    f2.close()
    f3.close()

combine_order_1_2(order_1_file,order_2_file)
