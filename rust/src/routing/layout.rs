use std::collections::HashMap;

use pyo3::{pyclass, pymethods};

#[derive(Clone)]
#[pyclass(module = "microboost.routing.sabre")]
pub(crate) struct MicroLayout {
    virt_to_phys: Vec<i32>,
    phys_to_virt: Vec<i32>,
}

#[pymethods]
impl MicroLayout {
    #[new]
    pub fn new(
        qubit_indices: HashMap<i32, i32>,
        virtual_qubits: usize,
        physical_qubits: usize,
    ) -> Self {
        let mut res = MicroLayout {
            virt_to_phys: vec![i32::MAX; virtual_qubits],
            phys_to_virt: vec![i32::MAX; physical_qubits],
        };
        for (virt, phys) in qubit_indices {
            res.virt_to_phys[virt as usize] = phys;
            res.phys_to_virt[phys as usize] = virt;
        }
        res
    }

    pub fn virtual_to_physical(&self, virt: i32) -> i32 {
        self.virt_to_phys[virt as usize]
    }

    pub fn physical_to_virtual(&self, phys: i32) -> i32 {
        self.phys_to_virt[phys as usize]
    }

    pub fn swap_virtual(&mut self, bit_a: i32, bit_b: i32) {
        self.virt_to_phys.swap(bit_a as usize, bit_b as usize);
        self.phys_to_virt[self.virt_to_phys[bit_a as usize] as usize] = bit_a;
        self.phys_to_virt[self.virt_to_phys[bit_b as usize] as usize] = bit_b;
    }

    pub fn swap_physical(&mut self, bit_a: i32, bit_b: i32) {
        self.phys_to_virt.swap(bit_a as usize, bit_b as usize);
        self.virt_to_phys[self.phys_to_virt[bit_a as usize] as usize] = bit_a;
        self.virt_to_phys[self.phys_to_virt[bit_b as usize] as usize] = bit_b;
    }
}
