r"""给定数独模板，自动算出结果。
调用库：sudoku.py from py-sudoku
使用方法：python pyshudu.py <输入数独模板文件名>
Example：python pyshudu.py puzzle.txt
输入文件要求：文件为文本文件，其中 # 后面是注释，
用逗号',' 隔开不同的数字。
以下含有9个数的数独模板 puzzle.txt
0, 0, 0, 0, 4, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 8
0, 0, 2, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 7, 0
0, 0, 0, 3, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 6, 0, 0
9, 0, 0, 0, 0, 0, 0, 0, 0
0, 1, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 5, 0, 0, 0

"""

from sys import argv
import time
import os
from numpy import loadtxt
from random import randrange, shuffle, sample  #, seed as rndseed
from math import floor
from sudoku.sudoku import Sudoku

# 小矩阵的长width=0   
# 小矩阵的宽height=0  
# 小矩阵的倍数blocks=0  


def read_board(files):
    '''用 numpy的 loadtxt 直接读取文本文件，创建二维数组ndarray，并转化为含整数的二维列表list'''

    _lst = loadtxt(files, comments="#", delimiter=",", dtype=int, \
                   encoding = 'utf-8', unpack=False).tolist()
    # 用了encoding='utf-8' 后，文件里可以出现中文注释。
    return _lst


def random_shudu(width=None, height=None):
    '''随机生成一组有效解的数独二维矩阵
    (TODO)最好改成 矩阵运算，用二维矩阵来定义比较好。
    '''

    if not height:
        height = width
    _size = width * height

    assert width >= 1,  "给定数独长度至少为1"
    assert height >= 1, "给定数独宽度至少为1"
    assert _size >= 1,  "给定数阵不能为 1 x 1"

    _a = [_ for _ in range(1, _size+1)]
    shuffle(_a)  # 随机打乱1~_size的顺序，_a自动变化
    _b = _a[1:] + _a[:1]  # 交换顺序
    _c = _a[2:] + _a[:2]  # 交换顺序

    _base =[_a, _a[width:] + _a[:width], _a[width*2:] + _a[:width*2],
            _b, _b[width:] + _b[:width], _b[width*2:] + _b[:width*2],
            _c, _c[width:] + _c[:width], _c[width*2:] + _c[:width*2]]
    return _base
    

#@staticmethod
def copy_board(board):
    return [[cell for cell in row] for row in board]

def random_puzzle(sudoku, difficulty=None):
    if not difficulty:
        difficulty = 0.9
    
    _shudu = copy_board(sudoku)
    _side = len(_shudu)
    _squares = _side * _side
    empty_cells = floor(_squares * difficulty)

    #难度系数算法如下：（key problem) 测试对3*3 没有问题，2*3有问题
    for p in sample(range(_squares), empty_cells):
        _shudu[p // _side][p % _side] = 0
        # 隐藏部分cells, 类Sudoku中接受的是 None
    return  _shudu


def puzzle_prt(puzzle):
    '''将puzzle中的 None 用整数 0 代替，方便打印输出'''
    return [[0 if not element else element for element in row] for \
             row in puzzle]


def show(files):
    '''编写自己的打印程序，并将题目和结果保存在数据文件中。
    '''
    pass


def test_Sudoku(m, n=None, difficulty=None):
    '''根据给定行数、列数和难度系数自动创建数独模板
    '''
    if n is None:
        n = m
    
    # given with difficulty (difficulty*100% of cells empty)
    if difficulty is not None:
        puzzle = Sudoku(m, n, difficulty)
    else:
        puzzle = Sudoku(m, n)

    puzzle.show_full()
    puzzle.solve().show_full()
    #puzzle = Sudoku(3, 3).difficulty(0.8)  # 接近1，表示空格多，难度大
    #puzzle = Sudoku(3, 3).difficulty(0.2)  # 接近0，表示空格少，难度低


def random_diff():
    '''随机产生难度系数，限制在[0.2,0.9] 之间'''
    return randrange(20, 90)/100


def reshape_puzzle(puzzle):
    '''重新排列二维列表，按照行列，如矩阵重新排列'''
    row_col = ""
    for row in puzzle:
        row_col = ''.join([row_col, '\n', str(row)]) 
    return row_col


def rand_puzzle(row, col=None, difficulty=0.9, outfile=None):
    '''测试自定义难度系数: 难度系数确省值为 0.9（难度大）'''

    if col is None:
        col = row

    # 随机产生一个正确的数独样例
    rndPuzzle = random_shudu(row, col)
    # 打印出来供参考，与实际结果对照，往往不一致，有多解。
    print(reshape_puzzle(rndPuzzle))
    # 提供难度系数后，随机隐藏某些cell，数字改为0
    puzzle = random_puzzle(rndPuzzle, difficulty)

    # 调用数独模块的类Sudoku，根据给定模板来创建一个实例数独 sudou
    sudou = Sudoku(row, col, board=puzzle)
    write_shudu(sudou, outfile)    


def helpme():
    '''显示帮助文件，介绍使用方法。'''
    print("""
    使用方法：python pyshudu.py <输入数独模板文件名>
    Example：python pyshudu.py puzzle.txt
    输入文件要求：文件为文本文件，其中 # 后面是注释，用逗号',' 隔开不同的数字。譬如

    # 以下是一个含有9个数字的数独模板文件 puzzle.txt
    0, 0, 0, 0, 4, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 0, 0, 8
    0, 0, 2, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 0, 7, 0
    0, 0, 0, 3, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 6, 0, 0
    9, 0, 0, 0, 0, 0, 0, 0, 0
    0, 1, 0, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 5, 0, 0, 0
    """)


def input_board( infile, outfile ):
    # given sample board, read from input file
    inboard = read_board(infile)

    row = len(inboard)
    col = len(inboard[0])
    # 此处 3 是否可以根据具体情况改成 最大公约数（或非1的真因子）。
    blocks = col//3  

    # 调用库函数中的类 Sudoku
    puzzle = Sudoku(blocks, board = inboard)
    write_shudu(puzzle, outfile)


def write_shudu(sudou, outfile=None):

    # 调用数独模块的类Sudoku，根据给定模板来创建一个实例数独 sudou
    sudou.show_full()  # 以矩阵形式显示 puzzle

    # 调用类的方法解题
    solution = sudou.solve()
    # 显示解题结果：有解，无解，或无效
    solution.show_full()

    if outfile is not None:
        size = sudou.width * sudou.height
        enter_html = '<div STYLE="page-break-after: always;"></div>\n\n'
        # 保留数独解题过程到文件中
        with open(outfile, mode='a+', encoding='utf-8') as outf:
            outf.write("## 数独案例 "+ time.asctime()+ "\n")
            outf.write(str(sudou)+'\n')
            outf.write(f"### 数阵有解 {size}X{size}：\n")
            outf.write(str(solution)+'\n')
            # 写入分页符 Markdown 格式中用的HTML分页符
            outf.write(enter_html)


if __name__ == '__main__':
    
    if len(argv) < 2:
        helpme()
        exit()
    
    input_fn = argv[1]
    
    if os.path.isfile(input_fn): 
        infile = input_fn
    else:
        print("输入的文件{0}不存在！".format(input_fn))
        exit()

    separator = ' '
    outfile = infile.split(sep='.')[0]
    if len(outfile) == 0:
        outfile = "tmp_shudu"
    outfile = ''.join([outfile, ".md"])

    answer = ''
    while answer.lower() != 'q':
        if answer.lower() == '/':
            input_board(infile, outfile)
        elif answer.lower() == '?':
            #测试随机产生数独序列
            rand_puzzle(3,3, difficulty=random_diff(), outfile=outfile)
        else:
            pass

        answer = input("q 退出, ? 随机出题, / 文件输入模板, 其他键继续: ")
