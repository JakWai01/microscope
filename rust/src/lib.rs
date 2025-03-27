use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyDict;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn micro_swap_boosted(py: Python, dag: PyObject, coupling_map: PyObject, initial_mapping:  &Bound<PyDict>) -> PyResult<PyObject> {
    let mut current_mapping = initial_mapping.copy()?;


    for (key, value) in current_mapping.iter() {
        println!("{} {}", key, value);
    }

    // println!("{:?}", current_mapping.get_item(0).unwrap().unwrap());

    
    let main_module = py.import("main")?; 
    let dag_class = main_module.getattr("DAG")?;
    let dag_instance = dag_class.call0()?;
    
    for node_id in 0..dag.call_method0(py, "__len__")?.extract::<i32>(py)? {
        let node = dag.call_method1(py, "get", (node_id,))?;
        
        let control = node.getattr(py, "control")?.extract::<i32>(py)?;
        let target = node.getattr(py, "target")?.extract::<i32>(py)?;
        
        let physical_q0 = current_mapping.get_item(control).unwrap().unwrap();
        let physical_q1 = current_mapping.get_item(target).unwrap().unwrap();

        // println!("{} {}", physical_q0, physical_q1);
        
        if coupling_map.call_method1(py, "distance", (&physical_q0, &physical_q1))?.extract::<i32>(py)? != 1 {
            // Returns the shortest undirected path between two physical qubits
            let path = coupling_map.call_method1(py, "shortest_undirected_path", (&physical_q0, &physical_q1))?.extract::<Vec<i32>>(py)?;
            
            for swap in 0..(path.len() - 2) {
                let connected_wire_1 = path[swap];
                let connected_wire_2 = path[swap + 1];
                
                // Check if we can improve the data sturcture to avoid this
                // Probably just maintaining both mapping is enough...
                let logical_q0 = get_logical_qubit(&current_mapping, connected_wire_1).unwrap();
                let logical_q1 = get_logical_qubit(&current_mapping, connected_wire_2).unwrap();
    
                let qubit_1 = current_mapping.get_item(logical_q0).unwrap().unwrap();
                let qubit_2 = current_mapping.get_item(logical_q1).unwrap().unwrap();

                let _ = dag_instance.call_method1("insert", (qubit_1, qubit_2, true));
            }

            for swap in 0..(path.len() - 2) {
                current_mapping = swap_physical_qubits(path[swap], path[swap + 1], current_mapping);
            }
        }

        let _ = dag_instance.call_method1("insert", (current_mapping.get_item(control).unwrap().unwrap(), current_mapping.get_item(target).unwrap().unwrap(), false));
    }
    
    Ok(dag_instance.into())
    // dag_instance.call_method1("insert", (12, 13, false))?;
    // let ress = dag_instance.call_method1("get", (0,))?;
    // println!("{:?}", ress.call_method0("__repr__")?.extract::<String>()?);

    // Ok(())
}

fn swap_physical_qubits(physical_q0: i32, physical_q1: i32, current_mapping: Bound<PyDict>) -> Bound<PyDict> {
    let logical_q0 = get_logical_qubit(&current_mapping, physical_q0).unwrap();
    let logical_q1 = get_logical_qubit(&current_mapping, physical_q1).unwrap();
    let tmp = current_mapping.get_item(logical_q0).unwrap().unwrap();
    let _ = current_mapping.set_item(logical_q0, current_mapping.get_item(logical_q1).unwrap().unwrap());
    let _ = current_mapping.set_item(logical_q1, tmp);
    current_mapping
}

fn get_logical_qubit(current_mapping: &Bound<PyDict>, connected_wire: i32) -> Option<i32> {
    for (key, value) in current_mapping.iter() {
        if value.extract::<i32>().unwrap() == connected_wire {
            return Some(key.extract::<i32>().unwrap())
        }
    }
    None
}

/// A Python module implemented in Rust.
#[pymodule]
fn microscope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(micro_swap_boosted, m)?)?;
    Ok(())
}
