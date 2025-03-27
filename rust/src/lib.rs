use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn micro_swap_boosted(py: Python, dag: PyObject, _coupling_map: PyObject, initial_mapping: PyObject) -> PyResult<()> {
    let current_mapping = initial_mapping.call_method1(py, "copy", ())?;
    
    let main_module = py.import("main")?; 
    
    let dag_class = main_module.getattr("DAG")?;
    let dag_instance = dag_class.call0()?;
    
    let node_args = (12, 13, false);
    dag_instance.call_method1("insert", node_args)?;
    let ress = dag_instance.call_method1("get", (0,))?;
    println!("{:?}", ress.call_method0("__repr__")?.extract::<String>()?);

    let args = (0,);
    let res = dag.call_method1(py, "get", args)?;
    println!("{:?}", res.call_method0(py, "__repr__")?.extract::<String>(py)?);
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn microscope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(micro_swap_boosted, m)?)?;
    Ok(())
}
