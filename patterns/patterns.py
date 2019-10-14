# coding=utf-8
#
# File : patterns.py
# Description : Library of 1-dimensional signals for testing.
# Date : 14th of October, 2019
#
# This file is part of the Conceptor package.  The Conceptor package is free
# software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>
#

# Import
import math
import numpy.random as random
import numpy as np

# Initializations
np.random.seed(8)

# 1 = sine10
# 2 = sine15
# 3 = sine20
# 4 = spike20
# 5 = spike10
# 6 = spike7
# 7 = 0
# 8 = 1
# 9 = rand4
# 10 = rand5
# 11 = rand6
# 12 = rand7
# 13 = rand8
# 14 = sine10range01
# 15 = sine10rangept5pt9
# 16 = rand3
# 17 = rand9
# 18 = rand10
# 19 = 0.8
# 20 = sineroot27
# 21 = sineroot19
# 22 = sineroot50
# 23 = sineroot75
# 24 = sineroot10
# 25 = sineroot110
# 26 = sineroot75tenth
# 27 = sineroots20plus40
# 28 = sineroot75third
# 29 = sineroot243
# 30 = sineroot150
# 31 = sineroot200
# 32 = sine10.587352723
# 33 = sine10.387352723
# 34 = rand7
# 35 = sine12
# 36 = 10+perturb
# 37 = sine11
# 38 = sine10.17352723
# 39 = sine5
# 40 = sine6
# 41 = sine7
# 42 = sine8
# 43 = sine9
# 44 = sine12
# 45 = sine13
# 46 = sine14
# 47 = sine10.8342522
# 48 = sine11.8342522
# 49 = sine12.8342522
# 50 = sine13.1900453
# 51 = sine7.1900453
# 52 = sine7.8342522
# 53 = sine8.8342522
# 54 = sine9.8342522
# 55 = sine5.19004
# 56 = sine5.8045
# 57 = sine6.49004
# 58 = sine6.9004
# 59 = sine13.9004
# 60 = 18+perturb
# 61 = spike3
# 62 = spike4
# 63 = spike5
# 64 = spike6
# 65 = rand4
# 66 = rand5
# 67 = rand6
# 68 = rand7
# 69 = rand8
# 70 = rand4
# 71 = rand5
# 72 = rand6
# 73 = rand7
# 74 = rand8

# List of patterns
patts = list()

# Pattern 1 to 8
patts.append(lambda n: math.sin(2 * math.pi * (n + 1) / 10))
patts.append(lambda n: math.sin(2 * math.pi * (n + 1) / 15))
patts.append(lambda n: math.sin(2 * math.pi * (n + 1) / 20))
patts.append(lambda n: int(1 == (n + 1) % 20))
patts.append(lambda n: int(1 == (n + 1) % 10))
patts.append(lambda n: int(1 == (n + 1) % 7))
patts.append(lambda n: 0)
patts.append(lambda n: 1)

# Pattern 9
rp_9 = np.array([-0.4564, 0.6712, -2.3953, -2.1594])
maxVal = np.max(rp_9)
minVal = np.min(rp_9)
rp_9 = 1.8 * (rp_9 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_9[(n + 1) % 4])

# Pattern 10
rp_10 = np.array([0.90155039, 0.51092795, 0.62290641, 0.20887359, 0.54710573])
maxVal = np.max(rp_10)
minVal = np.min(rp_10)
rp_10 = 1.8 * (rp_10 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_10[(n + 1) % 5])

# Pattern 11
rp_11 = np.array([0.5329, 0.9621, 0.1845, 0.5099, 0.3438, 0.7697])
maxVal = np.max(rp_11)
minVal = np.min(rp_11)
rp_11 = 1.8 * (rp_11 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_11[(n + 1) % 6])

# Pattern 12
rp_12 = np.array([0.8029, 0.4246, 0.2041, 0.0671, 0.1986, 0.2724, 0.5988])
maxVal = np.max(rp_12)
minVal = np.min(rp_12)
rp_12 = 1.8 * (rp_12 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_12[(n + 1) % 7])

# Pattern 13, 14 and 15
rp_13 = np.array([0.8731, 0.1282, 0.9582, 0.6832, 0.7420, 0.9829, 0.4161, 0.5316])
maxVal = np.max(rp_13)
minVal = np.min(rp_13)
rp_13 = 1.8 * (rp_13 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_13[(n + 1) % 8])
patts.append(lambda n: 0.5 * math.sin(2.0 * math.pi * n / 10.0) + 0.5)
patts.append(lambda n: 0.2 * math.sin(2.0 * math.pi * n / 10.0) + 0.7)

# Pattern 16
rp_16 = np.array([1.4101, -0.0992, -0.0902])
maxVal = np.max(rp_16)
minVal = np.min(rp_16)
rp_16 = 1.8 * (rp_16 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_16[(n + 1) % 3])

# Pattern 17
rp_17 = random.randn(9)
maxVal = np.max(rp_17)
minVal = np.min(rp_17)
rp_17 = 1.8 * (rp_17 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_17[n % 9])

# Pattern 18 to 33
rp_18 = random.randn(10)
maxVal = np.max(rp_18)
minVal = np.min(rp_18)
rp_18 = 1.8 * (rp_18 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_18[n % 10])
patts.append(lambda n: 0.8)
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(27)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(19)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(50)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(75)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(10)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(110)))
patts.append(lambda n: 0.1 * math.sin(2.0 * math.pi * n / math.sqrt(75)))
patts.append(lambda n: 0.5 * (math.sin(2.0 * math.pi * n / math.sqrt(20)) + math.sin(2 * math.pi * n / math.sqrt(40))))
patts.append(lambda n: 0.33 * math.sin(2.0 * math.pi * n / math.sqrt(75)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(243)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(150)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / math.sqrt(200)))
patts.append(lambda n: math.sin(2.0 * math.pi * n / 10.587352723))
patts.append(lambda n: math.sin(2 * math.pi * n / 10.387352723))

# Patterns 34 and 35
rp_34 = np.array([0.6792, 0.5129, 0.2991, 0.1054, 0.2849, 0.7689, 0.6408])
maxVal = np.max(rp_34)
minVal = np.min(rp_34)
rp_34 = 1.8 * (rp_34 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_34[(n + 1) % 7])
patts.append(lambda n: math.sin(2.0 * math.pi * n / 12.0))

# Pattern 36 to 59
rpDiff = np.array([-0.69209835, 0.7248628, -0.49427916, 1.16129582, 0.60450679])
rp_36 = rp_10 + 0.2 * rpDiff
# rp_36 = np.array([0.3419, 1.1282, 0.3107, -0.8949, -0.7266])
maxVal = np.max(rp_36)
minVal = np.min(rp_36)
rp_36 = 1.8 * (rp_36 - minVal) / (maxVal - minVal) - 0.9

patts.append(lambda n: rp_36[(n + 1) % 5])
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 11.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 10.17352723))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 5.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 6.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 7.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 8.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 9.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 12.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 13.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 14.0))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 10.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 11.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 12.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 13.1900453))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 7.1900453))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 7.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 8.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 9.8342522))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 5.1900453))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 5.804531))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 6.4900453))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 6.900453))
patts.append(lambda n: math.sin(2.0 * math.pi * (n + 1) / 13.900453))

# Pattern 60 to 64
rpDiff = random.randn(10)
rp_60 = rp_18 + 0.3 * rpDiff
maxVal = np.max(rp_60)
minVal = np.min(rp_60)
rp_60 = 1.8 * (rp_60 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_60[n % 10])
patts.append(lambda n: int(1 == n % 3))
patts.append(lambda n: int(1 == n % 4))
patts.append(lambda n: int(1 == n % 5))
patts.append(lambda n: int(1 == n % 6))

# Pattern 65
rp_65 = random.randn(4)
maxVal = np.max(rp_65)
minVal = np.min(rp_65)
rp_65 = 1.8 * (rp_65 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_65[n % 4])

# Pattern 66
rp_65 = random.rand(15)
maxVal = np.max(rp_65)
minVal = np.min(rp_65)
rp_65 = 1.8 * (rp_65 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_65[n % 5])

# Pattern 67
rp_67 = random.rand(6)
maxVal = np.max(rp_67)
minVal = np.min(rp_67)
rp_67 = 1.8 * (rp_67 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_67[n % 6])

# Pattern 68
rp_68 = random.rand(7)
maxVal = np.max(rp_68)
minVal = np.min(rp_68)
rp_68 = 1.8 * (rp_68 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_68[n % 7])

# Pattern 69
rp_69 = random.rand(8)
maxVal = np.max(rp_69)
minVal = np.min(rp_69)
rp_69 = 1.8 * (rp_69 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_69[n % 8])

# Pattern 70
rp_70 = random.randn(4)
maxVal = np.max(rp_70)
minVal = np.min(rp_70)
rp_70 = 1.8 * (rp_70 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_70[n % 4])

# Pattern 71
rp_71 = random.rand(5)
maxVal = np.max(rp_71)
minVal = np.min(rp_71)
rp_71 = 1.8 * (rp_71 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_71[n % 5])

# Pattern 72
rp_72 = random.rand(6)
maxVal = np.max(rp_72)
minVal = np.min(rp_72)
rp_72 = 1.8 * (rp_72 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_72[n % 6])

# Pattern 73
rp_73 = random.rand(7)
maxVal = np.max(rp_73)
minVal = np.min(rp_73)
rp_73 = 1.8 * (rp_73 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_73[n % 7])

# Pattern 74
rp_74 = random.rand(8)
maxVal = np.max(rp_74)
minVal = np.min(rp_74)
rp_74 = 1.8 * (rp_74 - minVal) / (maxVal - minVal) - 0.9
patts.append(lambda n: rp_74[n % 8])

# Pattern 75
random_period12 = np.array([1.0, -0.1, -1.0, 0.7, 0.85, -0.5, -0.1, 0.0, 1.0, -1.0])
max_val = np.max(random_period12)
min_val = np.min(random_period12)
random_period12 = 1.8 * (random_period12 - min_val) / (max_val - min_val) - 0.9
patts.append(lambda n: random_period12[(n + 1) % 10])

# Pattern 76
random_period13 = np.array([-1.0, 1.0, 0.9, 0.75, 0.55, 0.85, 1.0])
max_val = np.max(random_period13)
min_val = np.min(random_period13)
random_period13 = 1.8 * (random_period13 - min_val) / (max_val - min_val) - 0.9
patts.append(lambda n: random_period13[(n + 1) % 7])

# Pattern 77
random_period16 = np.array([-1.0, -0.25, 0.1, -0.32, 1.0, 0.85, 0.35])
max_val = np.max(random_period16)
min_val = np.min(random_period16)
random_period16 = 1.8 * (random_period16 - min_val) / (max_val - min_val) - 0.9
patts.append(lambda n: random_period16[(n + 1) % 7])
