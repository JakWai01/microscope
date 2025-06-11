use indexmap::IndexMap;
use pyo3::pyclass;

#[derive(Clone)]
#[pyclass(module = "microboost.routing.sabre")]
pub struct MicroFront {
    pub nodes: IndexMap<i32, [i32; 2]>,
    pub qubits: Vec<Option<(i32, i32)>>,
}

impl MicroFront {
    pub fn new(num_qubits: i32) -> Self {
        Self {
            nodes: IndexMap::with_capacity(num_qubits as usize / 2),
            qubits: vec![None; num_qubits as usize],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn insert(&mut self, index: i32, qubits: [i32; 2]) {
        let [a, b] = qubits;
        self.qubits[a as usize] = Some((index, b));
        self.qubits[b as usize] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    pub fn remove(&mut self, index: &i32) {
        let [a, b] = self
            .nodes
            .swap_remove(index)
            .expect("Tried removing index that does not exist.");
        self.qubits[a as usize] = None;
        self.qubits[b as usize] = None;
    }

    pub fn is_active(&self, qubit: i32) -> bool {
        self.qubits[qubit as usize].is_some()
    }

    pub fn apply_swap(&mut self, swap: [i32; 2]) {
        let [a, b] = swap;
        match (self.qubits[a as usize], self.qubits[b as usize]) {
            (Some((index1, _)), Some((index2, _))) if index1 == index2 => {
                let entry = self.nodes.get_mut(&index1).unwrap();
                *entry = [entry[1], entry[0]];
                return;
            }
            _ => {}
        }
        if let Some((index, c)) = self.qubits[a as usize] {
            self.qubits[c as usize] = Some((index, b));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [a, c] { [b, c] } else { [c, b] };
        }
        if let Some((index, c)) = self.qubits[b as usize] {
            self.qubits[c as usize] = Some((index, a));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [b, c] { [a, c] } else { [c, a] };
        }
        self.qubits.swap(a as usize, b as usize);
    }
}
