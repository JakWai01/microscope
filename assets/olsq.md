```
RUST_BACKTRACE=1 python3 src/microscope/main.py olsq
Trying maximal depth = 1...
Trying maximal depth = 2...
Trying maximal depth = 3...
Trying maximal depth = 4...
Trying maximal depth = 5...
Trying maximal depth = 6...
Trying maximal depth = 7...
Trying maximal depth = 8...
Trying maximal depth = 9...
Trying maximal depth = 10...
Trying maximal depth = 11...
Trying maximal depth = 12...
Trying maximal depth = 13...
Trying maximal depth = 14...
Trying maximal depth = 15...
Trying maximal depth = 16...
Compilation time = 4:13:06.120408.
SWAP on physical edge (1,2) at time 3
SWAP on physical edge (1,2) at time 4
SWAP on physical edge (2,3) at time 5
SWAP on physical edge (3,2) at time 6
SWAP on physical edge (3,4) at time 2
SWAP on physical edge (4,3) at time 7
SWAP on physical edge (4,5) at time 9
SWAP on physical edge (5,4) at time 8
SWAP on physical edge (6,5) at time 1
SWAP on physical edge (6,5) at time 10
SWAP on physical edge (6,7) at time 12
SWAP on physical edge (7,6) at time 11
SWAP on physical edge (7,8) at time 13
SWAP on physical edge (8,7) at time 0
SWAP on physical edge (8,9) at time 14
Gate 0: u3(pi,0,pi) 1 on qubit 8 at time 0
Gate 1: u3(pi,0,pi) 2 on qubit 7 at time 0
Gate 2: cx 1, 2 on qubits 8 and 7 at time 0
Gate 3: cx 1, 0 on qubits 8 and 9 at time 0
Gate 4: u3(pi/2,0,pi) 1 on qubit 8 at time 0
Gate 5: cx 2, 1 on qubits 7 and 8 at time 0
Gate 6: u3(0,0,-pi/4) 1 on qubit 8 at time 0
Gate 7: cx 0, 1 on qubits 9 and 8 at time 0
Gate 8: u3(0,0,pi/4) 1 on qubit 8 at time 0
Gate 9: cx 2, 1 on qubits 7 and 8 at time 0
Gate 10: u3(0,0,-pi/4) 1 on qubit 8 at time 0
Gate 11: cx 0, 1 on qubits 9 and 8 at time 0
Gate 12: u3(0,0,pi/4) 1 on qubit 7 at time 1
Gate 13: u3(pi/2,0,pi) 1 on qubit 7 at time 1
Gate 14: u3(0,0,pi/4) 2 on qubit 8 at time 11
Gate 15: cx 0, 2 on qubits 9 and 8 at time 11
Gate 16: u3(0,0,pi/4) 0 on qubit 9 at time 11
Gate 17: u3(0,0,-pi/4) 2 on qubit 8 at time 12
Gate 18: cx 0, 2 on qubits 9 and 8 at time 12
Gate 19: u3(pi,0,pi) 4 on qubit 5 at time 0
Gate 20: cx 3, 4 on qubits 6 and 5 at time 1
Gate 21: cx 3, 1 on qubits 6 and 7 at time 1
Gate 22: u3(pi/2,0,pi) 3 on qubit 6 at time 1
Gate 23: cx 4, 3 on qubits 5 and 6 at time 1
Gate 24: u3(0,0,-pi/4) 3 on qubit 6 at time 1
Gate 25: cx 1, 3 on qubits 7 and 6 at time 1
Gate 26: u3(0,0,pi/4) 3 on qubit 6 at time 1
Gate 27: cx 4, 3 on qubits 5 and 6 at time 1
Gate 28: u3(0,0,-pi/4) 3 on qubit 6 at time 1
Gate 29: cx 1, 3 on qubits 7 and 6 at time 1
Gate 30: u3(0,0,pi/4) 3 on qubit 5 at time 2
Gate 31: u3(pi/2,0,pi) 3 on qubit 5 at time 2
Gate 32: u3(0,0,pi/4) 4 on qubit 5 at time 1
Gate 33: cx 1, 4 on qubits 7 and 6 at time 2
Gate 34: u3(0,0,pi/4) 1 on qubit 7 at time 2
Gate 35: u3(0,0,-pi/4) 4 on qubit 6 at time 2
Gate 36: cx 1, 4 on qubits 7 and 6 at time 5
Gate 37: u3(pi,0,pi) 6 on qubit 3 at time 0
Gate 38: cx 5, 6 on qubits 4 and 3 at time 2
Gate 39: cx 5, 3 on qubits 4 and 5 at time 2
Gate 40: u3(pi/2,0,pi) 5 on qubit 4 at time 2
Gate 41: cx 6, 5 on qubits 3 and 4 at time 2
Gate 42: u3(0,0,-pi/4) 5 on qubit 4 at time 2
Gate 43: cx 3, 5 on qubits 5 and 4 at time 2
Gate 44: u3(0,0,pi/4) 5 on qubit 4 at time 2
Gate 45: cx 6, 5 on qubits 3 and 4 at time 2
Gate 46: u3(0,0,-pi/4) 5 on qubit 4 at time 2
Gate 47: cx 3, 5 on qubits 5 and 4 at time 2
Gate 48: u3(0,0,pi/4) 5 on qubit 4 at time 2
Gate 49: u3(pi/2,0,pi) 5 on qubit 3 at time 3
Gate 50: u3(0,0,pi/4) 6 on qubit 4 at time 4
Gate 51: cx 3, 6 on qubits 5 and 4 at time 4
Gate 52: u3(0,0,pi/4) 3 on qubit 5 at time 4
Gate 53: u3(0,0,-pi/4) 6 on qubit 4 at time 4
Gate 54: cx 3, 6 on qubits 5 and 4 at time 6
Gate 55: u3(pi,0,pi) 8 on qubit 1 at time 1
Gate 56: cx 7, 8 on qubits 2 and 1 at time 2
Gate 57: cx 7, 5 on qubits 2 and 3 at time 3
Gate 58: u3(pi/2,0,pi) 7 on qubit 2 at time 3
Gate 59: cx 8, 7 on qubits 1 and 2 at time 3
Gate 60: u3(0,0,-pi/4) 7 on qubit 2 at time 3
Gate 61: cx 5, 7 on qubits 3 and 2 at time 3
Gate 62: u3(0,0,pi/4) 7 on qubit 2 at time 3
Gate 63: cx 8, 7 on qubits 1 and 2 at time 3
Gate 64: u3(0,0,-pi/4) 7 on qubit 2 at time 3
Gate 65: cx 5, 7 on qubits 3 and 2 at time 3
Gate 66: u3(0,0,pi/4) 7 on qubit 2 at time 3
Gate 67: u3(pi/2,0,pi) 7 on qubit 2 at time 3
Gate 68: u3(0,0,pi/4) 8 on qubit 1 at time 3
Gate 69: cx 5, 8 on qubits 3 and 2 at time 4
Gate 70: u3(0,0,pi/4) 5 on qubit 3 at time 4
Gate 71: u3(0,0,-pi/4) 8 on qubit 2 at time 4
Gate 72: cx 5, 8 on qubits 3 and 2 at time 4
Gate 73: cx 7, 9 on qubits 1 and 0 at time 4
Gate 74: u3(pi/2,0,pi) 7 on qubit 1 at time 4
Gate 75: cx 8, 7 on qubits 2 and 1 at time 4
Gate 76: u3(0,0,-pi/4) 7 on qubit 1 at time 4
Gate 77: cx 5, 7 on qubits 3 and 2 at time 5
Gate 78: u3(0,0,pi/4) 7 on qubit 2 at time 5
Gate 79: cx 8, 7 on qubits 1 and 2 at time 5
Gate 80: u3(0,0,-pi/4) 7 on qubit 2 at time 5
Gate 81: cx 5, 7 on qubits 3 and 2 at time 5
Gate 82: u3(0,0,pi/4) 7 on qubit 2 at time 5
Gate 83: u3(pi/2,0,pi) 7 on qubit 3 at time 6
Gate 84: u3(0,0,pi/4) 8 on qubit 1 at time 5
Gate 85: cx 5, 8 on qubits 2 and 1 at time 6
Gate 86: u3(0,0,pi/4) 5 on qubit 2 at time 6
Gate 87: u3(0,0,-pi/4) 8 on qubit 1 at time 6
Gate 88: cx 5, 8 on qubits 2 and 1 at time 6
Gate 89: cx 7, 5 on qubits 3 and 2 at time 6
Gate 90: cx 5, 8 on qubits 2 and 1 at time 6
Gate 91: u3(pi/2,0,pi) 5 on qubit 2 at time 6
Gate 92: cx 6, 5 on qubits 4 and 3 at time 7
Gate 93: u3(0,0,-pi/4) 5 on qubit 3 at time 7
Gate 94: cx 3, 5 on qubits 5 and 4 at time 8
Gate 95: u3(0,0,pi/4) 5 on qubit 4 at time 8
Gate 96: cx 6, 5 on qubits 3 and 4 at time 8
Gate 97: u3(0,0,-pi/4) 5 on qubit 4 at time 8
Gate 98: cx 3, 5 on qubits 4 and 5 at time 9
Gate 99: u3(0,0,pi/4) 5 on qubit 5 at time 9
Gate 100: u3(pi/2,0,pi) 5 on qubit 5 at time 9
Gate 101: u3(0,0,pi/4) 6 on qubit 3 at time 8
Gate 102: cx 3, 6 on qubits 4 and 3 at time 9
Gate 103: u3(0,0,pi/4) 3 on qubit 4 at time 9
Gate 104: u3(0,0,-pi/4) 6 on qubit 3 at time 9
Gate 105: cx 3, 6 on qubits 4 and 3 at time 9
Gate 106: cx 5, 3 on qubits 5 and 4 at time 9
Gate 107: cx 3, 6 on qubits 4 and 3 at time 9
Gate 108: u3(pi/2,0,pi) 3 on qubit 4 at time 9
Gate 109: cx 4, 3 on qubits 6 and 5 at time 10
Gate 110: u3(0,0,-pi/4) 3 on qubit 5 at time 10
Gate 111: cx 1, 3 on qubits 7 and 6 at time 11
Gate 112: u3(0,0,pi/4) 3 on qubit 6 at time 11
Gate 113: cx 4, 3 on qubits 5 and 6 at time 11
Gate 114: u3(0,0,-pi/4) 3 on qubit 6 at time 11
Gate 115: cx 1, 3 on qubits 7 and 6 at time 11
Gate 116: u3(0,0,pi/4) 3 on qubit 6 at time 11
Gate 117: u3(pi/2,0,pi) 3 on qubit 6 at time 11
Gate 118: u3(0,0,pi/4) 4 on qubit 5 at time 11
Gate 119: cx 1, 4 on qubits 6 and 5 at time 12
Gate 120: u3(0,0,pi/4) 1 on qubit 6 at time 12
Gate 121: u3(0,0,-pi/4) 4 on qubit 5 at time 12
Gate 122: cx 1, 4 on qubits 6 and 5 at time 12
Gate 123: cx 3, 1 on qubits 7 and 6 at time 12
Gate 124: cx 1, 4 on qubits 6 and 5 at time 12
Gate 125: u3(pi/2,0,pi) 1 on qubit 6 at time 12
Gate 126: cx 2, 1 on qubits 8 and 7 at time 13
Gate 127: u3(0,0,-pi/4) 1 on qubit 8 at time 14
Gate 128: cx 0, 1 on qubits 9 and 8 at time 14
Gate 129: u3(0,0,pi/4) 1 on qubit 8 at time 14
Gate 130: cx 2, 1 on qubits 7 and 8 at time 14
Gate 131: u3(0,0,-pi/4) 1 on qubit 8 at time 14
Gate 132: cx 0, 1 on qubits 9 and 8 at time 14
Gate 133: u3(0,0,pi/4) 1 on qubit 8 at time 14
Gate 134: u3(pi/2,0,pi) 1 on qubit 8 at time 14
Gate 135: u3(0,0,pi/4) 2 on qubit 7 at time 14
Gate 136: cx 0, 2 on qubits 8 and 7 at time 15
Gate 137: u3(0,0,pi/4) 0 on qubit 8 at time 15
Gate 138: u3(0,0,-pi/4) 2 on qubit 7 at time 15
Gate 139: cx 0, 2 on qubits 8 and 7 at time 15
Gate 140: cx 1, 0 on qubits 9 and 8 at time 15
Gate 141: cx 0, 2 on qubits 8 and 7 at time 15
result additional SWAP count = 15.
Took 15186.24s
```