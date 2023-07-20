import os

# 不同学习率对模型性能影响 随机种子10， 0 ， 1， 2， 3
# os.system('python main.py --lr 0.1')
# os.system('python main.py --lr 0.01')
# os.system('python main.py --lr 0.001')
# os.system('python main.py --lr 0.0001')
# os.system('python main.py --lr 0.00001')

# 是否进行正则化
# os.system('python main.py --source1 .\BearingData\SN30\Condition0.mat --source2 .\BearingData\SN30\Condition1.mat '
#           '--source3 .\BearingData\SN30\Condition2.mat --source4 .\BearingData\SN30\Condition3.mat --weightdecay 0')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0' )
# os.system('python main.py --source1 .\BearingData\SN30\Condition2.mat --source2 .\BearingData\SN30\Condition3.mat '
#           '--source3 .\BearingData\SN30\Condition0.mat --source4 .\BearingData\SN30\Condition1.mat --weightdecay 0' )
# os.system('python main.py --source1 .\BearingData\SN30\Condition3.mat --source2 .\BearingData\SN30\Condition0.mat '
#           '--source3 .\BearingData\SN30\Condition1.mat --source4 .\BearingData\SN30\Condition2.mat --weightdecay 0' )

# 正则化参数讨论(轴承数据)  Mar13_16-24-34_DESKTOP-HRPP07F-Mar13_16-36-11_DESKTOP-HRPP07F
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0.1')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0.01')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0.001')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0.0001')

# 正则化参数讨论(齿轮数据)
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#           '--source3 .\GearData\Fea\Condition3.mat --source4 .\GearData\Fea\Condition0.mat --weightdecay 0')
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#           '--source3 .\GearData\Fea\Condition3.mat --source4 .\GearData\Fea\Condition0.mat --weightdecay 0.1')
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#           '--source3 .\GearData\Fea\Condition3.mat --source4 .\GearData\Fea\Condition0.mat --weightdecay 0.01')
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#           '--source3 .\GearData\Fea\Condition3.mat --source4 .\GearData\Fea\Condition0.mat --weightdecay 0.001')
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#           '--source3 .\GearData\Fea\Condition3.mat --source4 .\GearData\Fea\Condition0.mat --weightdecay 0.0001')

####   是否进行分布权重优化  BS412-BT3  轴承随机种子 10，5，130，120，50
os.system('python main.py --lr 0.01 --random_seed 10')
os.system('python main.py --lr 0.01 --random_seed 5')
os.system('python main.py --lr 0.01 --random_seed 130')
os.system('python main.py --lr 0.01 --random_seed 120')
os.system('python main.py --lr 0.01 --random_seed 50')

###   混淆矩阵
# os.system('python main.py --source1 .\GearData\Fea\Condition1.mat --source2 .\GearData\Fea\Condition2.mat '
#            '--source3 .\GearData\Fea\Condition3.mat --source3 .\GearData\Fea\Condition0.mat --weightdecay 0.01')
# os.system('python main.py --source1 .\BearingData\SN30\Condition1.mat --source2 .\BearingData\SN30\Condition2.mat '
#           '--source3 .\BearingData\SN30\Condition3.mat --source4 .\BearingData\SN30\Condition0.mat --weightdecay 0.1')