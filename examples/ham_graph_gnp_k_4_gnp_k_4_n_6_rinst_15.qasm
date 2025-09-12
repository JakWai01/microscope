OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate gate_PauliEvolution(param0) q0,q1,q2,q3,q4,q5 { rxx(-0.5) q4,q5; rxx(-0.5) q3,q5; rxx(-0.5) q2,q5; rxx(-0.5) q1,q5; rxx(-0.5) q0,q5; ryy(-0.5) q4,q5; ryy(-0.5) q3,q5; ryy(-0.5) q2,q5; ryy(-0.5) q1,q5; ryy(-0.5) q0,q5; rzz(-0.5) q4,q5; rzz(-0.5) q3,q5; rzz(-0.5) q2,q5; rzz(-0.5) q1,q5; rzz(-0.5) q0,q5; rxx(-0.5) q3,q4; rxx(-0.5) q2,q4; rxx(-0.5) q1,q4; rxx(-0.5) q0,q4; ryy(-0.5) q3,q4; ryy(-0.5) q2,q4; ryy(-0.5) q1,q4; ryy(-0.5) q0,q4; rzz(-0.5) q3,q4; rzz(-0.5) q2,q4; rzz(-0.5) q1,q4; rzz(-0.5) q0,q4; rxx(-0.5) q2,q3; rxx(-0.5) q1,q3; rxx(-0.5) q0,q3; ryy(-0.5) q2,q3; ryy(-0.5) q1,q3; ryy(-0.5) q0,q3; rzz(-0.5) q2,q3; rzz(-0.5) q1,q3; rzz(-0.5) q0,q3; rxx(-0.5) q1,q2; rxx(-0.5) q0,q2; ryy(-0.5) q1,q2; ryy(-0.5) q0,q2; rzz(-0.5) q1,q2; rzz(-0.5) q0,q2; rxx(-0.5) q0,q1; ryy(-0.5) q0,q1; rzz(-0.5) q0,q1; }
qreg q[6];
gate_PauliEvolution(1.0) q[0],q[1],q[2],q[3],q[4],q[5];
